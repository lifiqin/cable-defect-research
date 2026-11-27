"""Local Faster R-CNN R50-FPN-v2 baseline for the same group split.

Uses a different standard training recipe from Ultralytics. It defaults to
Ultralytics box matching/AP; the optional COCO backend requires torchmetrics.
Cross-framework comparisons are exploratory, not controlled module ablations.
"""

import argparse
import json
import os
import random
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from cable_research.faster_neck_head import modify_faster
from cable_research.split import create_split


def decode_yolo(lines, width, height):
    boxes, labels = [], []
    for line in lines:
        row = line.split()
        if len(row) != 5 or row[0] not in {"0", "1"}:
            raise ValueError("invalid detection label")
        x, y, w, h = map(float, row[1:])
        if not (0 < w <= 1 and 0 < h <= 1 and -1e-6 <= x - w / 2 and
                x + w / 2 <= 1 + 1e-6 and -1e-6 <= y - h / 2 and y + h / 2 <= 1 + 1e-6):
            raise ValueError("invalid box geometry")
        box = ((x - w / 2) * width, (y - h / 2) * height,
               (x + w / 2) * width, (y + h / 2) * height)
        box = (max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3]))
        if not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError("invalid box geometry")
        boxes.append(box)
        labels.append(int(row[0]) + 1)  # torchvision reserves class 0 for background.
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(labels, dtype=torch.int64)


def collate(batch):
    return tuple(zip(*batch, strict=True))


def is_better(validation, best, selection):
    if selection not in {"map50", "map50_95"}:
        raise ValueError("unsupported checkpoint selection metric")
    return float(validation[selection]) > best


class ScratchDataset(Dataset):
    def __init__(self, image_list: Path, dataset: Path, size: int, augment: bool):
        self.paths = [Path(s) for s in image_list.read_text(encoding="utf-8").splitlines() if s]
        self.dataset, self.size, self.augment = dataset, size, augment
        if not self.paths:
            raise ValueError("empty split")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
        image.thumbnail((self.size, self.size), Image.Resampling.BILINEAR)
        width, height = image.size
        label = self.dataset / "labels/train" / f"{path.stem}.txt"
        boxes, labels = decode_yolo(label.read_text(encoding="utf-8").splitlines(), width, height)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(boxes):
                left = width - boxes[:, 2].clone()
                boxes[:, 2] = width - boxes[:, 0]
                boxes[:, 0] = left
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([index]),
                  "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                  "iscrowd": torch.zeros(len(labels), dtype=torch.int64)}
        return pil_to_tensor(image).float() / 255, target


class UltralyticsAP:
    """Apply the same Ultralytics box matching/AP implementation to torchvision outputs."""

    def __init__(self):
        self.matcher = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
        self.tp, self.conf, self.pred_cls, self.target_cls = [], [], [], []

    def update(self, pred, target):
        from ultralytics.engine.validator import BaseValidator
        from ultralytics.utils.metrics import box_iou

        pred = {k: v.detach().cpu() for k, v in pred.items()}
        target = {k: v.detach().cpu() for k, v in target.items()}
        labels, classes = target["labels"], pred["labels"]
        if len(labels) and len(classes):
            iou = box_iou(target["boxes"], pred["boxes"])
            correct = BaseValidator.match_predictions(self.matcher, classes, labels, iou).numpy()
        else:
            correct = np.zeros((len(classes), 10), dtype=bool)
        self.tp.append(correct)
        self.conf.append(pred["scores"].numpy())
        self.pred_cls.append(classes.numpy())
        self.target_cls.append(labels.numpy())

    def compute(self):
        from ultralytics.utils.metrics import ap_per_class

        if not self.target_cls or not any(len(target) for target in self.target_cls):
            raise ValueError("validation split contains no ground-truth objects")
        values = ap_per_class(np.concatenate(self.tp), np.concatenate(self.conf), np.concatenate(self.pred_cls),
                              np.concatenate(self.target_cls), plot=False)
        precision, recall, ap, classes = values[2], values[3], values[5], values[6]
        return {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean()),
                "class_map50_95": {str(int(c)): float(score.mean())
                                   for c, score in zip(classes, ap, strict=True)},
                "class_ap50": {str(int(c)): float(score[0]) for c, score in zip(classes, ap, strict=True)},
                "class_precision": {str(int(c)): float(p) for c, p in zip(classes, precision, strict=True)},
                "class_recall": {str(int(c)): float(r) for c, r in zip(classes, recall, strict=True)}}


def evaluate(model, loader, device, backend="ultralytics"):
    if backend == "ultralytics":
        metric = UltralyticsAP()
        model.eval()
        with torch.inference_mode():
            for images, targets in loader:
                predictions = model([x.to(device) for x in images])
                for prediction, target in zip(predictions, targets, strict=True):
                    metric.update(prediction, target)
        return metric.compute()
    if backend != "coco":
        raise ValueError("unsupported metric backend")
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True,
                                  max_detection_thresholds=[1, 10, 300], backend="faster_coco_eval")
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            predictions = model([x.to(device) for x in images])
            metric.update([{k: v.cpu() for k, v in p.items()} for p in predictions],
                          [{k: t[k].cpu() for k in ("boxes", "labels", "area", "iscrowd")} for t in targets])
    result = metric.compute()
    classes = {str(int(cls)): float(ap) for cls, ap in zip(result["classes"], result["map_per_class"], strict=True)}
    return {"map50": float(result["map_50"]), "map50_95": float(result["map"]), "class_map50_95": classes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--fold-plan", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--fold", type=int, choices=(1, 2, 3), default=2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neck", choices=("standard", "context"), default="standard")
    ap.add_argument("--head", choices=("standard", "taskdual"), default="standard")
    ap.add_argument("--metric", choices=("ultralytics", "coco"), default="ultralytics")
    ap.add_argument("--selection", choices=("map50", "map50_95"), default="map50_95")
    a = ap.parse_args()
    if not 1 <= a.epochs <= 300 or a.eval_every < 1 or a.imgsz < 128 or a.batch < 1:
        ap.error("invalid budget")
    dataset, weights, work = a.dataset.resolve(strict=True), a.weights.resolve(strict=True), a.work.resolve()
    if work.exists():
        raise FileExistsError("refusing to overwrite an experiment")
    _, counts = create_split(dataset, a.fold_plan, work / "split", a.fold)
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["WANDB_DISABLED"] = "true"

    def offline(*args, **kwargs):
        raise OSError("network disabled during private-data training")

    socket.create_connection = offline
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=a.imgsz, max_size=a.imgsz,
                                      box_score_thresh=0.001, box_detections_per_img=300)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 3)
    model = modify_faster(model, neck=a.neck, head=a.head)
    for name, param in model.backbone.body.named_parameters():
        param.requires_grad = name.startswith(("layer2", "layer3", "layer4"))
    model.to(device)
    optim = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.005, momentum=0.9,
                            weight_decay=0.0005)
    schedule = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[max(1, int(a.epochs * .7)),
                                                                    max(2, int(a.epochs * .9))], gamma=0.1)
    train = ScratchDataset(work / "split/train.txt", dataset, a.imgsz, augment=True)
    val = ScratchDataset(work / "split/val.txt", dataset, a.imgsz, augment=False)
    train_loader = DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    best = -1.0
    (work / "protocol.json").write_text(json.dumps({"counts": counts, "model": "FasterRCNN R50-FPN-v2",
        "training_recipe": "SGD 0.005, horizontal flip, evaluation every N epochs; not YOLO recipe",
        "neck": a.neck, "head": a.head, "metric": a.metric, "selection": a.selection,
        "classes": {"1": "scratch", "2": "hole"},
        "eval_max_dets": 300}, indent=2) + "\n", encoding="utf-8")
    with (work / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, a.epochs + 1):
            model.train()
            total = 0.0
            for images, targets in train_loader:
                loss = sum(model([x.to(device) for x in images],
                                 [{k: v.to(device) for k, v in t.items()} for t in targets]).values())
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite detection loss")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                total += float(loss.detach())
            schedule.step()
            row = {"epoch": epoch, "mean_train_loss": total / len(train_loader)}
            if epoch == 1 or epoch % a.eval_every == 0 or epoch == a.epochs:
                row["validation"] = evaluate(model, val_loader, device, backend=a.metric)
                if is_better(row["validation"], best, a.selection):
                    best = row["validation"][a.selection]
                    torch.save(model.state_dict(), work / "best_state.pt")
                    (work / "best_validation.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    print("Completed Faster R-CNN baseline", flush=True)


if __name__ == "__main__":
    main()
