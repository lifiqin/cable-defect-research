"""Run one local-only grouped-validation experiment from a local checkpoint."""

import argparse
import os
import socket
from pathlib import Path

from cable_research.mixer import attach_mixer
from cable_research.split import create_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--fold-plan", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True, help="local trusted YOLO11n checkpoint")
    ap.add_argument("--work", type=Path, required=True, help="new private local output directory")
    ap.add_argument("--mode", choices=("baseline", "isotropic", "uniform", "tensor", "contrast"), required=True)
    ap.add_argument("--fold", type=int, choices=(1, 2, 3), default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    if not 1 <= a.epochs <= 300 or a.batch < 1 or a.imgsz < 128:
        ap.error("invalid training budget")
    weights = a.weights.resolve(strict=True)
    work = a.work.resolve()
    if work.exists():
        raise FileExistsError("refusing to overwrite an existing experiment")
    data, counts = create_split(a.dataset, a.fold_plan, work / "split", a.fold)
    print("Approximate grouped train/validation:", counts, flush=True)
    os.environ["YOLO_CONFIG_DIR"] = str(work / "ultralytics_config")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["WANDB_DISABLED"] = "true"

    def offline(*args, **kwargs):
        raise OSError("network disabled during local-only data training")

    socket.create_connection = offline
    socket.socket.connect = offline
    import torch
    from ultralytics import YOLO
    from ultralytics.data import utils as data_utils
    from ultralytics.models.yolo.detect import DetectionTrainer

    data_utils.check_font = lambda *args, **kwargs: None
    torch.set_num_threads(4)

    class ResearchTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            return attach_mixer(model, a.mode)

    model = YOLO(weights)
    model.train(trainer=ResearchTrainer, data=str(data), epochs=a.epochs, imgsz=a.imgsz,
                batch=a.batch, workers=0, device=a.device, project=str(work), name="run",
                exist_ok=False, seed=a.seed, deterministic=True, amp=False, plots=False,
                optimizer="AdamW", lr0=0.002, close_mosaic=0, patience=0, verbose=False,
                cache="ram", val=True, save=True)
    print("Completed", a.mode, "in", work, flush=True)


if __name__ == "__main__":
    main()
