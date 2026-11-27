"""Record validation results without creating a trained-model checkpoint."""

import json
import math

from cable_research.evaluate import metrics_summary


class NoCheckpointMixin:
    """Keep training and validation, but never serialize the trained model."""

    def save_model(self):
        return False

    def final_eval(self):
        # The fit callback already collected every epoch's validation results.
        return None


class EpochMetricsCapture:
    def __init__(self, work):
        self.work = work
        self.path = work / "validation_epochs.jsonl"
        self.best = None
        # Fail closed: a fresh experiment must not append to an older result.
        self.path.touch(exist_ok=False)

    def on_fit_epoch_end(self, trainer):
        fitness = float(trainer.fitness)
        if not math.isfinite(fitness):
            raise ValueError("non-finite validation fitness")
        result = metrics_summary(trainer.validator.metrics.box, trainer.data["names"])
        result.update(epoch=int(trainer.epoch) + 1, fitness=fitness)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, allow_nan=False) + "\n")
        # Ultralytics chooses the most recent epoch on a fitness tie.
        if self.best is None or fitness >= self.best["fitness"]:
            self.best = result

    def finish(self, imgsz, batch):
        if self.best is None:
            raise RuntimeError("no completed validation epochs")
        result = dict(self.best)
        result.update(imgsz=imgsz, batch=batch, model_retained=False,
                      selection="maximum Ultralytics validation fitness; latest tie",
                      evaluation="training-time EMA validation, not a saved-checkpoint re-evaluation",
                      purpose="validation only; not an independent test or a SOTA claim")
        with (self.work / "metrics.json").open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
        return result
