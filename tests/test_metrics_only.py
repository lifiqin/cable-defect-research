import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cable_research.metrics_only import EpochMetricsCapture, NoCheckpointMixin


class MetricsOnlyTests(unittest.TestCase):
    def trainer(self, epoch, fitness):
        box = SimpleNamespace(ap_class_index=(0, 1), ap50=(0.3, 0.7), ap=(0.2, 0.5),
                              p=(0.4, 0.8), r=(0.6, 0.9), map50=0.5, map=0.35)
        return SimpleNamespace(epoch=epoch, fitness=fitness, data={"names": {0: "scratch", 1: "hole"}},
                               validator=SimpleNamespace(metrics=SimpleNamespace(box=box)))

    def test_records_each_epoch_and_uses_latest_fitness_tie(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            capture = EpochMetricsCapture(work)
            for epoch, fitness in enumerate((0.3, 0.2, 0.3)):
                capture.on_fit_epoch_end(self.trainer(epoch, fitness))
            result = capture.finish(640, 2)
            self.assertEqual(result["epoch"], 3)
            self.assertEqual(result["classes"]["scratch"]["ap50"], 0.3)
            self.assertFalse(result["model_retained"])
            self.assertEqual(len((work / "validation_epochs.jsonl").read_text().splitlines()), 3)
            self.assertEqual(json.loads((work / "metrics.json").read_text())["map50"], 0.5)
            self.assertFalse(list(work.rglob("*.pt")))

    def test_refuses_overwrite_and_missing_or_invalid_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            capture = EpochMetricsCapture(work)
            with self.assertRaises(FileExistsError):
                EpochMetricsCapture(work)
            with self.assertRaises(RuntimeError):
                capture.finish(640, 2)
            with self.assertRaises(ValueError):
                capture.on_fit_epoch_end(self.trainer(0, float("nan")))

    def test_checkpoint_hooks_write_nothing(self):
        self.assertFalse(NoCheckpointMixin().save_model())
        self.assertIsNone(NoCheckpointMixin().final_eval())


if __name__ == "__main__":
    unittest.main()
