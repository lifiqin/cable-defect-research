import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from cable_research.train import main


class TrainOptionTests(unittest.TestCase):
    def test_invalid_recipe_parameters_are_rejected_before_io(self):
        required = ["train", "--dataset", "unused", "--fold-plan", "unused", "--weights", "unused",
                    "--work", "unused", "--mode", "baseline"]
        for option in (["--nbs", "0"], ["--warmup-bias-lr", "2"], ["--mosaic", "1.1"],
                       ["--family", "yolo", "--neck", "geometry_context"],
                       ["--metrics-only", "--head", "p2"], ["--metrics-only", "--device", "0,1"]):
            with (self.subTest(option=option), patch.object(sys, "argv", required + option),
                  contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
                main()


if __name__ == "__main__":
    unittest.main()
