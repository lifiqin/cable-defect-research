"""Two-stage FPN and task-head adapters; not a reproduction of DPN-Detector."""

import torch
from torch import nn


class ContextFPN(nn.Module):
    """Retain pretrained FPN and add zero-initialized context on P2 and P3."""

    def __init__(self, base: nn.Module, channels: int = 256):
        super().__init__()
        self.base = base
        self.blocks = nn.ModuleDict({key: nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(),
        ) for key in ("0", "1")})
        self.alpha = nn.ParameterDict({key: nn.Parameter(torch.tensor(0.0)) for key in self.blocks})

    def forward(self, x):
        values = self.base(x)
        if any(k not in values for k in self.blocks):
            raise ValueError("expected torchvision FPN levels 0 and 1")
        for key, block in self.blocks.items():
            values[key] = values[key] + self.alpha[key] * block(values[key])
        return values


class TaskDecoupledPredictor(nn.Module):
    """Retain the predictor while giving classification and regression separate adapters."""

    def __init__(self, base: nn.Module):
        super().__init__()
        if not isinstance(getattr(base, "cls_score", None), nn.Linear) or not isinstance(
                getattr(base, "bbox_pred", None), nn.Linear):
            raise TypeError("expected a torchvision FastRCNNPredictor")
        channels = base.cls_score.in_features
        hidden = max(32, channels // 8)
        self.base = base
        self.class_adapter = self._adapter(channels, hidden)
        self.box_adapter = self._adapter(channels, hidden)

    @staticmethod
    def _adapter(channels, hidden):
        result = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels))
        nn.init.zeros_(result[-1].weight)
        nn.init.zeros_(result[-1].bias)
        return result

    def forward(self, x):
        if x.dim() == 4 and list(x.shape[2:]) != [1, 1]:
            raise ValueError("expected pooled 1x1 predictor input")
        x = x.flatten(start_dim=1)
        return self.base.cls_score(x + self.class_adapter(x)), self.base.bbox_pred(x + self.box_adapter(x))


def modify_faster(model, neck="standard", head="standard", seed=34567):
    if neck not in {"standard", "context"} or head not in {"standard", "taskdual"}:
        raise ValueError("unsupported Faster R-CNN neck/head")
    if (not hasattr(getattr(model, "backbone", None), "fpn") or
            not hasattr(getattr(model, "roi_heads", None), "box_predictor")):
        raise ValueError("expected a torchvision Faster R-CNN FPN model")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if neck == "context":
            model.backbone.fpn = ContextFPN(model.backbone.fpn)
        if head == "taskdual":
            model.roi_heads.box_predictor = TaskDecoupledPredictor(model.roi_heads.box_predictor)
    return model
