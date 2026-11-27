import copy
import unittest

import torch
from torch import nn

from cable_research.neck_head import ContextDown, TaskGatedBranch, WeightedConcat, modify_rtdetr, modify_yolo


class FakeConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.i, self.f, self.type = 12, [-1, 6], "Concat"


class NeckHeadTests(unittest.TestCase):
    def test_fusion_is_identity_initialized_and_differentiable(self):
        x = [torch.randn(2, 3, 8, 8, requires_grad=True), torch.randn(2, 5, 8, 8, requires_grad=True)]
        expected = torch.cat(x, dim=1)
        for mode in ("weighted", "adaptive"):
            with self.subTest(mode=mode):
                module = WeightedConcat(FakeConcat(), mode)
                actual = module(x)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual((module.i, module.f), (12, [-1, 6]))
                actual.square().mean().backward(retain_graph=True)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_task_branch_initial_output_is_unchanged(self):
        branch = nn.Sequential(nn.Conv2d(4, 7, 3, padding=1), nn.ReLU(), nn.Conv2d(7, 2, 1))
        wrapped = TaskGatedBranch(branch)
        x = torch.randn(2, 4, 10, 10)
        self.assertTrue(torch.equal(branch(x), wrapped(x)))
        wrapped(x).sum().backward()
        self.assertIsNotNone(wrapped.gate[-1].weight.grad)

    def test_context_down_preserves_pretrained_branch_at_initialization(self):
        class FakeDown(nn.Module):
            def __init__(self):
                super().__init__()
                self.i, self.f, self.type = 22, -1, "Conv"
                self.conv = nn.Conv2d(6, 6, 3, stride=2, padding=1)

            def forward(self, x):
                return self.conv(x)

        block = FakeDown().eval()
        context = ContextDown(block).eval()
        x = torch.randn(2, 6, 16, 16)
        self.assertTrue(torch.equal(block(x), context(x)))
        context(x).sum().backward()
        self.assertIsNotNone(context.alpha.grad)

    def test_yolo11_neck_and_task_head_keep_initial_predictions(self):
        from ultralytics import YOLO

        torch.set_num_threads(2)
        model = YOLO("yolo11n.yaml").model.eval()
        original = copy.deepcopy(model).eval()
        before = torch.random.get_rng_state().clone()
        modify_yolo(model, neck="weighted", head="taskgate")
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        x = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            a, b = original(x)[0], model(x)[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-6))

    def test_unexpected_rtdetr_topology_is_rejected(self):
        class Fake(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.ModuleList([nn.Identity() for _ in range(29)])

        with self.assertRaisesRegex(ValueError, "topology"):
            modify_rtdetr(Fake(), neck="context")


if __name__ == "__main__":
    unittest.main()
