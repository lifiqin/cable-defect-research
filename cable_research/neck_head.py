"""Ablatable neck and task-head variants, inspired by published mechanisms.

These are deliberately simple research implementations, not exact reproductions
of BiFPN, DyHead or ContextGuidedDown. Retain pretrained layers and initialize
the modified graph to the original network's output.
"""

import torch
from torch import nn


class WeightedConcat(nn.Module):
    """Preserve concat topology while learning static or per-image input weights."""

    def __init__(self, old: nn.Module, mode: str):
        super().__init__()
        if mode not in {"weighted", "adaptive"}:
            raise ValueError("unsupported neck mode")
        self.i, self.f = old.i, old.f
        self.type = f"{old.type}+{mode}"
        self.mode = mode
        if mode == "weighted":
            self.logits = nn.Parameter(torch.zeros(2))
        else:
            self.router = nn.Linear(4, 2)
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("weighted fusion requires exactly two feature inputs")
        if self.mode == "weighted":
            weights = (2 * self.logits.softmax(dim=0)).view(1, 2)
        else:
            features = torch.stack([t.mean((1, 2, 3)) for t in x] +
                                   [t.abs().mean((1, 2, 3)) for t in x], dim=1)
            weights = 2 * self.router(features).softmax(dim=1)
        return torch.cat([t * weights[:, i].view(-1, 1, 1, 1) for i, t in enumerate(x)], dim=1)


class TaskGatedBranch(nn.Module):
    """An independently learned, identity-initialized gate before one head branch."""

    def __init__(self, branch: nn.Module):
        super().__init__()
        conv = next((m for m in branch.modules() if isinstance(m, nn.Conv2d)), None)
        if conv is None:
            raise ValueError("cannot infer head branch input channels")
        channels = conv.in_channels
        hidden = max(8, min(channels // 8, 64))
        self.branch = branch
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.SiLU(),
                                  nn.Conv2d(hidden, channels, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x):
        return self.branch(x * (1 + 0.25 * self.gate(x).tanh()))


class ContextDown(nn.Module):
    """RT-DETR neck-downsampling adapter with local and dilated context paths."""

    def __init__(self, old: nn.Module):
        super().__init__()
        if not hasattr(old, "conv") or old.conv.stride != (2, 2):
            raise ValueError("expected a stride-2 pretrained neck convolution")
        channels = old.conv.out_channels
        self.i, self.f, self.type = old.i, old.f, f"{old.type}+ContextDown"
        self.base = old
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.context = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False)
        self.mix = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU())
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        base = self.base(x)
        update = self.mix(torch.cat((self.local(base), self.context(base)), dim=1))
        return base + self.alpha * update


def modify_yolo(model, neck="none", head="standard", seed=23456):
    """Replace known YOLO11 fusion nodes and/or wrap its decoupled branches."""
    if neck not in {"none", "weighted", "adaptive"} or head not in {"standard", "taskgate"}:
        raise ValueError("unsupported YOLO neck or head")
    if neck == "none" and head == "standard":
        return model
    layers = model.model
    if len(layers) != 24:
        raise ValueError("expected the standard 24-layer YOLO11 graph")
    indices = (12, 15, 18, 21)
    if neck != "none" and any(getattr(layers[i], "i", None) != i or
                              layers[i].__class__.__name__ != "Concat" for i in indices):
        raise ValueError("unexpected neck topology")
    detect = layers[-1]
    if head == "taskgate" and (detect.__class__.__name__ != "Detect" or detect.nl != 3 or detect.end2end):
        raise ValueError("expected a standard non-end-to-end three-scale YOLO11 head")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if neck != "none":
            for index in indices:
                layers[index] = WeightedConcat(layers[index], neck)
        if head == "taskgate":
            for branches in (detect.cv2, detect.cv3):
                for index, branch in enumerate(branches):
                    branches[index] = TaskGatedBranch(branch)
    return model


def modify_rtdetr(model, neck="none", seed=23456):
    """Wrap known RT-DETR-L bottom-up neck downsampling nodes, not its decoder."""
    if neck not in {"none", "context"}:
        raise ValueError("unsupported RT-DETR neck")
    if neck == "none":
        return model
    layers, indices = model.model, (22, 25)
    if len(layers) != 29 or any(getattr(layers[i], "i", None) != i or
                                not hasattr(layers[i], "conv") or layers[i].conv.stride != (2, 2)
                                for i in indices):
        raise ValueError("unexpected RT-DETR-L neck topology")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for index in indices:
            layers[index] = ContextDown(layers[index])
    return model
