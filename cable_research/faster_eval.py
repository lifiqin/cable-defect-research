"""Re-evaluate a locally trained Faster R-CNN state dictionary on a local split."""

import argparse
import json
import os
import socket
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cable_research.faster_neck_head import modify_faster
from cable_research.faster_rcnn import ScratchDataset, collate, evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True, help="private validation image-list file")
    ap.add_argument("--weights", type=Path, required=True, help="trusted local best_state.pt")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--neck", choices=("standard", "context"), default="standard")
    ap.add_argument("--head", choices=("standard", "taskdual"), default="standard")
    ap.add_argument("--metric", choices=("ultralytics", "coco"), default="ultralytics")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = ap.parse_args()
    dataset, split, weights = (x.resolve(strict=True) for x in (a.dataset, a.split, a.weights))
    work = a.work.resolve()
    if work.exists() or a.imgsz < 128:
        raise ValueError("invalid output directory or image size")
    work.mkdir(parents=True)
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["WANDB_DISABLED"] = "true"

    def offline(*args, **kwargs):
        raise OSError("network disabled during private-data evaluation")

    socket.create_connection = offline
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    torch.set_num_threads(3)
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=3,
                                      min_size=a.imgsz, max_size=a.imgsz, box_score_thresh=0.001,
                                      box_detections_per_img=300)
    model = modify_faster(model, neck=a.neck, head=a.head)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    device = torch.device(a.device)
    model.to(device)
    data = ScratchDataset(split, dataset, a.imgsz, augment=False)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    result = {"purpose": "validation only; not an independent test", "neck": a.neck, "head": a.head,
              "metric": a.metric, "images": len(data), "validation": evaluate(model, loader, device, a.metric)}
    (work / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
