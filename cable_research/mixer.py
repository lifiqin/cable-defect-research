"""Local-orientation-conditioned feature mixing for controlled experiments.

This is a proposed research prototype, not a claim of novelty or SOTA. The
structure tensor is computed from the current feature map, never from labels.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class DiagonalDepthwise(nn.Module):
    """A depthwise diagonal filter with k, rather than k squared, weights."""

    def __init__(self, channels: int, kernel_size: int = 7, antidiagonal: bool = False):
        super().__init__()
        if channels <= 0 or kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError("channels and odd kernel_size must be positive")
        self.kernel_size = kernel_size
        self.antidiagonal = antidiagonal
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = torch.diag_embed(self.weight).unsqueeze(1)
        if self.antidiagonal:
            kernel = kernel.flip(-1)
        return F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=x.shape[1])


def structure_components(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Return bounded double-angle components of a smoothed structure tensor.

    Components approach zero for weak or isotropic structure; their magnitude
    also controls confidence in direction rather than dividing by tiny norms.
    """
    if window < 3 or window % 2 != 1:
        raise ValueError("structure window must be an odd integer of at least 3")
    gray = x.mean(dim=1, keepdim=True)
    kx = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8
    ky = kx.transpose(-1, -2)
    gx, gy = F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)
    jxx = F.avg_pool2d(gx.square(), window, stride=1, padding=window // 2)
    jyy = F.avg_pool2d(gy.square(), window, stride=1, padding=window // 2)
    jxy = F.avg_pool2d(gx * gy, window, stride=1, padding=window // 2)
    energy = jxx + jyy + 1e-6
    return torch.cat(((jxx - jyy) / energy, 2 * jxy / energy), dim=1)


class GeometryMixer(nn.Module):
    """Ablatable identity-residual feature mixer.

    ``isotropic`` uses a square depthwise convolution, ``uniform`` averages
    four directional experts, ``tensor`` weights them using local structure,
    ``contrast`` uses local-minus-context structure, and ``hybrid`` combines
    local tensor routing with a square-kernel residual branch.
    """

    def __init__(self, channels: int, mode: str, kernel_size: int = 7, channel_gate: bool = False):
        super().__init__()
        if mode not in {"isotropic", "uniform", "tensor", "contrast", "hybrid"}:
            raise ValueError(f"unsupported mixer mode: {mode}")
        self.mode = mode
        if mode in {"isotropic", "hybrid"}:
            self.square = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2,
                                    groups=channels, bias=False)
        if mode != "isotropic":
            self.horizontal = nn.Conv2d(channels, channels, (1, kernel_size),
                                        padding=(0, kernel_size // 2), groups=channels, bias=False)
            self.vertical = nn.Conv2d(channels, channels, (kernel_size, 1),
                                      padding=(kernel_size // 2, 0), groups=channels, bias=False)
            self.diagonal = DiagonalDepthwise(channels, kernel_size)
            self.antidiagonal = DiagonalDepthwise(channels, kernel_size, antidiagonal=True)
        if mode in {"tensor", "contrast", "hybrid"}:
            self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        if channel_gate:
            hidden = max(8, channels // 8)
            self.channel_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                              nn.SiLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        else:
            self.channel_gate = None
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "isotropic":
            mixed = self.square(x)
        else:
            experts = torch.stack((self.horizontal(x), self.vertical(x), self.diagonal(x),
                                   self.antidiagonal(x)), dim=1)
            if self.mode == "uniform":
                mixed = experts.mean(dim=1)
            else:
                uv = structure_components(x)
                if self.mode == "contrast":
                    uv = uv - structure_components(x, window=21)
                u, v = uv[:, :1], uv[:, 1:]
                logits = torch.cat((-u, u, -v, v), dim=1) * F.softplus(self.log_temperature)
                weights = logits.softmax(dim=1).unsqueeze(2)
                mixed = (experts * weights).sum(dim=1)
            if self.mode == "hybrid":
                mixed = (mixed + self.square(x)) * 0.5
        update = self.project(mixed)
        gate = getattr(self, "channel_gate", None)  # Older trusted checkpoints predate this optional field.
        if gate is not None:
            update = update * gate(x)
        return x + self.alpha * update


class WrappedFeatureBlock(nn.Module):
    """Preserve YOLO graph metadata while augmenting one pretrained block."""

    def __init__(self, block: nn.Module, mixer: nn.Module):
        super().__init__()
        self.block, self.mixer = block, mixer
        self.i, self.f = block.i, block.f
        self.type = f"{block.type}+GeometryMixer"
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.mixer(self.block(x))


def attach_mixer(model: nn.Module, mode: str, seed: int = 12345, sites: str = "p3",
                 channel_gate: bool = False) -> nn.Module:
    """Attach at the selected P3/P4 head outputs after pretrained loading."""
    if sites not in {"p3", "p3p4"}:
        raise ValueError("unsupported feature sites")
    if mode == "baseline":
        return model
    indices = (16,) if sites == "p3" else (16, 19)
    if len(model.model) <= max(indices):
        raise ValueError("unexpected YOLO11 graph; refusing to alter an unknown model")
    blocks = [model.model[i] for i in indices]
    if any(getattr(b, "i", None) != i or not hasattr(b, "cv2") for i, b in zip(indices, blocks, strict=True)):
        raise ValueError("unexpected YOLO11 graph; refusing to alter an unknown model")
    for offset, (index, block) in enumerate(zip(indices, blocks, strict=True)):
        channels = block.cv2.conv.out_channels
        # Preserve the baseline RNG state and the same P3 initialization across sites.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + offset)
            mixer = GeometryMixer(channels, mode, channel_gate=channel_gate)
        model.model[index] = WrappedFeatureBlock(block, mixer)
    return model
