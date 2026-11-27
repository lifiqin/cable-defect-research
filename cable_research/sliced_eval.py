"""Full-image versus overlap-tiled validation using original image annotations.

This is a simple local slicing baseline, not a reproduction of the SAHI or
SCAHI algorithms. Tiles are created only at inference; the validation labels
are never cropped or used to decide tile locations.
"""

import argparse
import json
import os
import socket
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import batched_nms

from cable_research.faster_rcnn import UltralyticsAP, decode_yolo


def starts(length, tile, overlap):
    if length < 1 or tile < 1 or not 0 <= overlap < 0.8:
        raise ValueError("invalid tiling geometry")
    limit = max(length - tile, 0)
    step = max(1, int(tile * (1 - overlap)))
    result = list(range(0, limit + 1, step))
    if result[-1] != limit:
        result.append(limit)
    return result


def windows(width, height, tile, overlap):
    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in starts(height, tile, overlap) for x in starts(width, tile, overlap)]


def combine(predictions, nms_iou, max_dets):
    if not predictions:
        return {"boxes": torch.empty((0, 4)), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.int64)}
    boxes = torch.cat([p["boxes"] for p in predictions])
    scores = torch.cat([p["scores"] for p in predictions])
    labels = torch.cat([p["labels"] for p in predictions])
    keep = batched_nms(boxes, scores, labels, nms_iou)[:max_dets]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


def extract(result, x, y):
    boxes = result.boxes.xyxy.detach().cpu().float().clone()
    boxes[:, 0::2] += x
    boxes[:, 1::2] += y
    return {"boxes": boxes, "scores": result.boxes.conf.detach().cpu().float(),
            "labels": result.boxes.cls.detach().cpu().long() + 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--mode", choices=("full", "tile", "both"), default="full")
    ap.add_argument("--tile", type=int, default=2048, help="tile side in original-image pixels")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--augment", action="store_true", help="Ultralytics augmented inference")
    ap.add_argument("--nms-iou", type=float, default=0.7)
    a = ap.parse_args()
    dataset, split, weights = (p.resolve(strict=True) for p in (a.dataset, a.split, a.weights))
    work = a.work.resolve()
    if (work.exists() or a.imgsz < 128 or a.tile < 128 or a.batch < 1 or not 0 <= a.overlap < 0.8 or
            not 0 < a.nms_iou <= 1):
        raise ValueError("invalid output or inference configuration")
    paths = [Path(line).resolve(strict=True) for line in split.read_text(encoding="utf-8").splitlines() if line]
    source_dir = (dataset / "images/train").resolve(strict=True)
    if not paths or any(p.parent != source_dir or p.suffix.lower() != ".jpg" for p in paths):
        raise ValueError("split must reference original dataset training images only")
    work.mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(work / "ultralytics_config")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["WANDB_DISABLED"] = "true"

    def offline(*args, **kwargs):
        raise OSError("network disabled during private-data evaluation")

    socket.create_connection = offline
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    from ultralytics import YOLO

    from cable_research import mixer, neck_head  # noqa: F401

    torch.set_num_threads(3)
    model = YOLO(weights)
    metric = UltralyticsAP()
    passes = 0
    common = {"imgsz": a.imgsz, "conf": 0.001, "iou": a.nms_iou, "max_det": 300,
              "device": a.device, "verbose": False, "augment": a.augment}
    for index, path in enumerate(paths, 1):
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        targets, labels = decode_yolo((dataset / "labels/train" / f"{path.stem}.txt").read_text(
            encoding="utf-8").splitlines(), width, height)
        predictions = []
        if a.mode in {"full", "both"}:
            predictions.append(extract(model.predict(image, **common)[0], 0, 0))
            passes += 1
        if a.mode in {"tile", "both"}:
            crops = windows(width, height, a.tile, a.overlap)
            for offset in range(0, len(crops), a.batch):
                batch = crops[offset:offset + a.batch]
                results = model.predict([image.crop(box) for box in batch], **common)
                predictions.extend(extract(result, box[0], box[1]) for result, box in zip(results, batch, strict=True))
                passes += len(batch)
        metric.update(combine(predictions, nms_iou=a.nms_iou, max_dets=300), {"boxes": targets, "labels": labels})
        if index % 10 == 0 or index == len(paths):
            print(f"Evaluated {index}/{len(paths)} images", flush=True)
    result = {"purpose": "approximate-group validation only; not an independent test", "mode": a.mode,
              "tile_original_pixels": a.tile if a.mode != "full" else None, "overlap": a.overlap,
              "imgsz": a.imgsz, "augment": a.augment, "images": len(paths), "model_passes": passes,
              "merge": f"class-aware NMS at {a.nms_iou}, maximum 300 per original image",
              "validation": metric.compute()}
    (work / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
