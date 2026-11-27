import unittest

import torch
from torch import nn

from cable_research.mixer import (
    DiagonalDepthwise,
    GeometryMixer,
    attach_mixer,
    structure_components,
)


class MixerTests(unittest.TestCase):
    def test_flat_tensor_is_finite_and_has_no_preferred_orientation(self):
        components = structure_components(torch.zeros(2, 4, 11, 13))
        self.assertTrue(torch.isfinite(components).all())
        self.assertEqual(torch.count_nonzero(components).item(), 0)

    def test_structure_flips_axes_when_image_is_transposed(self):
        x = torch.arange(21.0).view(1, 1, 1, 21).expand(1, 2, 21, 21)
        a = structure_components(x)[..., 4:-4, 4:-4]
        b = structure_components(x.transpose(-1, -2))[..., 4:-4, 4:-4]
        self.assertTrue(torch.allclose(a[:, 0], -b[:, 0], atol=1e-5))
        self.assertTrue(torch.allclose(a[:, 1], b[:, 1], atol=1e-5))

    def test_uniform_ramp_has_negligible_local_context_contrast(self):
        x = torch.arange(41.0).view(1, 1, 1, 41).expand(1, 2, 41, 41)
        contrast = structure_components(x, 5) - structure_components(x, 21)
        self.assertLess(contrast[..., 12:-12, 12:-12].abs().max().item(), 1e-5)

    def test_mixer_modes_preserve_shape_and_backpropagate(self):
        for mode in ("isotropic", "uniform", "tensor", "contrast"):
            with self.subTest(mode=mode):
                model = GeometryMixer(6, mode)
                x = torch.randn(2, 6, 15, 17, requires_grad=True)
                y = model(x)
                self.assertEqual(x.shape, y.shape)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_diagonal_depthwise_has_only_k_weights_per_channel(self):
        layer = DiagonalDepthwise(3, kernel_size=7)
        self.assertEqual(layer.weight.numel(), 21)
        self.assertEqual(layer(torch.ones(1, 3, 9, 9)).shape, (1, 3, 9, 9))

    def test_attach_preserves_rng_and_graph_metadata(self):
        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.cv2 = nn.Module()
                self.cv2.conv = nn.Conv2d(4, 4, 1)
                self.i, self.f, self.type = 16, -1, "FakeBlock"

            def forward(self, x):
                return self.cv2.conv(x)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.ModuleList([nn.Identity() for _ in range(16)] + [FakeBlock()])

        model = FakeModel()
        before = torch.random.get_rng_state().clone()
        old = model.model[16]
        attach_mixer(model, "tensor")
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertIs(model.model[16].block, old)
        self.assertEqual((model.model[16].i, model.model[16].f), (16, -1))
        self.assertEqual(model.model[16](torch.randn(1, 4, 7, 7)).shape, (1, 4, 7, 7))


if __name__ == "__main__":
    unittest.main()
