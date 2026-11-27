"""Evaluate local YOLO, RT-DETR and Faster R-CNN ensembles on original images.

This is inference-time NMS ensembling, not a new network architecture. Metrics
remain development-validation results and must not be called an independent test.
"""

import argparse
import json
import os
import socket
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from cable_research.faster_neck_head import modify_faster
from cable_research.faster_rcnn import UltralyticsAP, decode_yolo
from cable_research.sliced_eval import combine, extract


def scale_boxes(boxes, original, resized):
    boxes = boxes.detach().cpu().float().clone()
    boxes[:, 0::2] *= original[0] / resized[0]
    boxes[:, 1::2] *= original[1] / resized[1]
    return boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--yolo", type=Path, action="append", default=[])
    ap.add_argument("--rtdetr", type=Path, action="append", default=[])
    ap.add_argument("--faster", type=Path, help="local Faster R-CNN best_state.pt")
    ap.add_argument("--faster-neck", choices=("standard", "context"), default="standard")
    ap.add_argument("--faster-head", choices=("standard", "taskdual"), default="standard")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nms-iou", type=float, default=0.45)
    ap.add_argument("--augment", action="store_true", help="apply YOLO test-time augmentation")
    a = ap.parse_args()
    if (len(a.yolo) + len(a.rtdetr) + int(a.faster is not None) < 2 or a.work.exists() or a.imgsz < 128 or
            not 0 < a.nms_iou <= 1):
        raise ValueError("need at least two local models and a new output directory")
    dataset, split = a.dataset.resolve(strict=True), a.split.resolve(strict=True)
    yolo_weights = [p.resolve(strict=True) for p in a.yolo]
    rtdetr_weights = [p.resolve(strict=True) for p in a.rtdetr]
    faster_weights = a.faster.resolve(strict=True) if a.faster else None
    source_dir = (dataset / "images/train").resolve(strict=True)
    paths = [Path(s).resolve(strict=True) for s in split.read_text(encoding="utf-8").splitlines() if s]
    if not paths or any(p.parent != source_dir or p.suffix.lower() != ".jpg" for p in paths):
        raise ValueError("split must reference original dataset training images only")
    work = a.work.resolve()
    work.mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(work / "ultralytics_config")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["WANDB_DISABLED"] = "true"

    def offline(*args, **kwargs):
        raise OSError("network disabled during private-data evaluation")

    socket.create_connection = offline
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    from ultralytics import RTDETR, YOLO

    from cable_research import mixer, neck_head  # noqa: F401

    torch.set_num_threads(3)
    yolos = [YOLO(p) for p in yolo_weights]
    rtdetrs = [RTDETR(p) for p in rtdetr_weights]
    if any(model.names != {0: "scratch", 1: "hole"} for model in yolos + rtdetrs):
        raise ValueError("ensemble members must have scratch=0 and hole=1 class names")
    faster = None
    if faster_weights:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

        faster = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=3,
                                            min_size=a.imgsz, max_size=a.imgsz, box_score_thresh=0.001,
                                            box_detections_per_img=300)
        faster = modify_faster(faster, neck=a.faster_neck, head=a.faster_head)
        faster.load_state_dict(torch.load(faster_weights, map_location="cpu", weights_only=True), strict=True)
        faster.eval()
    metric = UltralyticsAP()
    for index, path in enumerate(paths, 1):
        with Image.open(path) as source:
            image = source.convert("RGB")
        gt, labels = decode_yolo((dataset / "labels/train" / f"{path.stem}.txt").read_text(
            encoding="utf-8").splitlines(), *image.size)
        predictions = []
        for model in yolos:
            result = model.predict(image, device="cpu", imgsz=a.imgsz, conf=0.001, iou=a.nms_iou,
                                   max_det=300, verbose=False, augment=a.augment)[0]
            predictions.append(extract(result, 0, 0))
        for model in rtdetrs:
            result = model.predict(image, device="cpu", imgsz=a.imgsz, conf=0.001,
                                   max_det=300, verbose=False)[0]
            predictions.append(extract(result, 0, 0))
        if faster is not None:
            resized = image.copy()
            resized.thumbnail((a.imgsz, a.imgsz), Image.Resampling.BILINEAR)
            with torch.inference_mode():
                prediction = faster([pil_to_tensor(resized).float() / 255])[0]
            prediction["boxes"] = scale_boxes(prediction["boxes"], image.size, resized.size)
            predictions.append(prediction)
        metric.update(combine(predictions, a.nms_iou, 300), {"boxes": gt, "labels": labels})
        if index % 10 == 0 or index == len(paths):
            print(f"Evaluated {index}/{len(paths)} images", flush=True)
    result = {"purpose": "development validation only; not an independent test", "yolo_count": len(yolos),
              "rtdetr_count": len(rtdetrs), "faster_count": int(faster is not None),
              "imgsz": a.imgsz, "yolo_tta": a.augment,
              "nms_iou": a.nms_iou, "images": len(paths), "validation": metric.compute()}
    (work / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
