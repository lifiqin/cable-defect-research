"""Evaluate a trusted checkpoint locally without uploading data or labels."""

import argparse
import json
import os
import socket
from pathlib import Path


def metrics_summary(box, names):
    classes = {}
    for i, cls in enumerate(box.ap_class_index):
        key = int(cls)
        classes[str(names[key])] = {"ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i]),
                                    "precision": float(box.p[i]), "recall": float(box.r[i])}
    return {"map50": float(box.map50), "map50_95": float(box.map), "classes": classes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--family", choices=("yolo", "rtdetr"), default="yolo")
    a = ap.parse_args()
    weights, data, work = a.weights.resolve(strict=True), a.data.resolve(strict=True), a.work.resolve()
    if work.exists():
        raise FileExistsError("refusing to overwrite evaluation output")
    work.mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(work / "ultralytics_config")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["WANDB_DISABLED"] = "true"

    def offline(*args, **kwargs):
        raise OSError("network disabled for local evaluation")

    socket.create_connection = offline
    socket.socket.connect = offline
    from ultralytics import RTDETR, YOLO
    from ultralytics.data import utils as data_utils

    from cable_research import mixer  # noqa: F401

    data_utils.check_font = lambda *args, **kwargs: None
    model = YOLO(weights) if a.family == "yolo" else RTDETR(weights)
    metrics = model.val(data=str(data), split="val", imgsz=a.imgsz, batch=a.batch, device=a.device,
                        workers=0, plots=False, verbose=False, save_json=False,
                        project=str(work), name="evaluation", exist_ok=False)
    result = metrics_summary(metrics.box, model.names)
    result["purpose"] = "validation only; this is not an independent test or a SOTA claim"
    result["imgsz"], result["batch"] = a.imgsz, a.batch
    with (work / "metrics.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
