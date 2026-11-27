import unittest
from collections import OrderedDict

import torch
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from cable_research.faster_neck_head import ContextFPN, TaskDecoupledPredictor, modify_faster


class IdentityFPN(nn.Module):
    def forward(self, x):
        return OrderedDict((k, v) for k, v in x.items())


class FasterNeckHeadTests(unittest.TestCase):
    def test_fpn_context_is_identity_with_gradient(self):
        neck = ContextFPN(IdentityFPN(), channels=8).eval()
        x = OrderedDict((key, torch.randn(2, 8, 8, 8)) for key in ("0", "1", "2"))
        out = neck(x)
        for key in x:
            self.assertTrue(torch.equal(out[key], x[key]))
        sum(y.square().mean() for y in out.values()).backward()
        self.assertIsNotNone(neck.alpha["0"].grad)
        with self.assertRaisesRegex(ValueError, "levels"):
            neck({"2": x["2"]})

    def test_task_predictor_is_identity_with_gradient(self):
        base = FastRCNNPredictor(16, 3)
        head = TaskDecoupledPredictor(base)
        x = torch.randn(4, 16)
        before, after = base(x), head(x)
        for expected, actual in zip(before, after, strict=True):
            self.assertTrue(torch.equal(expected, actual))
        sum(y.square().mean() for y in after).backward()
        self.assertIsNotNone(head.class_adapter[-1].weight.grad)
        self.assertIsNotNone(head.box_adapter[-1].weight.grad)

    def test_unexpected_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "FPN"):
            modify_faster(nn.Identity(), neck="context")


if __name__ == "__main__":
    unittest.main()
