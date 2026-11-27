"""Evaluate a trusted local RF-DETR checkpoint on the private validation split."""

import argparse
import json
import os
import socket
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--max-dets", type=int, default=300)
    a = ap.parse_args()
    dataset, weights, work = a.dataset_dir.resolve(strict=True), a.weights.resolve(strict=True), a.work.resolve()
    if work.exists() or not (dataset / "data.yaml").is_file() or a.max_dets < 10:
        raise ValueError("invalid dataset, output, or maximum detections")
    work.mkdir(parents=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ.pop("ROBOFLOW_API_KEY", None)

    def offline(*args, **kwargs):
        raise OSError("network disabled during private-data evaluation")

    socket.create_connection = offline
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    import torch
    from rfdetr import RFDETR

    torch.set_num_threads(3)
    model = RFDETR.from_checkpoint(weights, trust_checkpoint=True, device=a.device)
    result = model.evaluate(split="val", dataset_dir=str(dataset), output_dir=str(work), device=a.device,
                            batch_size=1, num_workers=0, log_per_class_metrics=True, eval_max_dets=a.max_dets,
                            tensorboard=False, wandb=False)
    payload = {"purpose": "validation, not an independent test", "max_detections": a.max_dets,
               "metrics": result}
    (work / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
