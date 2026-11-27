import unittest

import torch
from torch import nn

from cable_research.benchmark import measure


class BenchmarkTests(unittest.TestCase):
    def test_cpu_latency_is_positive(self):
        result = measure(nn.Identity(), torch.zeros(1, 3, 4, 4), warmup=1, trials=3)
        self.assertGreater(result["median_ms"], 0)
        self.assertEqual(result["trials"], 3)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            measure(nn.Identity(), torch.zeros(1), warmup=0, trials=1)


if __name__ == "__main__":
    unittest.main()
