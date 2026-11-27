import unittest

import torch

from cable_research.ensemble_eval import scale_boxes


class EnsembleTests(unittest.TestCase):
    def test_rescale_to_original_image(self):
        boxes = torch.tensor([[5., 10., 25., 50.]])
        scaled = scale_boxes(boxes, (400, 800), (100, 200))
        self.assertTrue(torch.equal(scaled, torch.tensor([[20., 40., 100., 200.]])))
        self.assertTrue(torch.equal(boxes, torch.tensor([[5., 10., 25., 50.]])))
        self.assertEqual(scale_boxes(torch.empty(0, 4), (400, 800), (100, 200)).shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
