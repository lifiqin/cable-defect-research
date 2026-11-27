"""Validate an approximate group partition before creating local manifests."""

import json
import re
from pathlib import Path

import yaml

STEM = re.compile(r"^[A-Za-z0-9_-]+$")


def create_split(dataset: Path, plan_file: Path, out: Path, held_out_fold: int):
    if held_out_fold not in (1, 2, 3):
        raise ValueError("held_out_fold must be 1, 2, or 3")
    dataset, plan_file, out = dataset.resolve(strict=True), plan_file.resolve(strict=True), out.resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    folds = plan["folds"]
    if len(folds) != 3 or any(not isinstance(fold, list) for fold in folds):
        raise ValueError("three fold lists are required")
    stems = [s for fold in folds for s in fold]
    if any(not isinstance(s, str) or not STEM.fullmatch(s) for s in stems):
        raise ValueError("unsafe image stem")
    if len(stems) != len(set(stems)):
        raise ValueError("image appears in more than one fold")
    ready = [Path(line.strip()).stem for line in (dataset / "train_ready.txt").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(ready) != len(set(ready)) or set(ready) != set(stems):
        raise ValueError("fold membership does not exactly match train_ready.txt")
    fold_of = {stem: i for i, fold in enumerate(folds, 1) for stem in fold}
    for group in plan["groups"]:
        if any(fold_of.get(stem) != group["fold"] for stem in group["stems"]):
            raise ValueError("a group straddles folds or names an unknown image")
    grouped = [stem for group in plan["groups"] for stem in group["stems"]]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(stems):
        raise ValueError("groups do not partition all images exactly once")
    for stem in stems:
        if not (dataset / "images" / "train" / f"{stem}.jpg").is_file():
            raise FileNotFoundError(f"missing training image: {stem}")
        if not (dataset / "labels" / "train" / f"{stem}.txt").is_file():
            raise FileNotFoundError(f"missing training label: {stem}")
    if out.exists():
        raise FileExistsError("refusing to overwrite a prepared split")
    counts, manifests = {}, {}
    for name in ("train", "val"):
        chosen = [s for s in stems if (fold_of[s] != held_out_fold) == (name == "train")]
        paths = [(dataset / "images" / "train" / f"{s}.jpg").as_posix() for s in chosen]
        manifests[name] = paths
        boxes = [0, 0]
        for stem in chosen:
            for line in (dataset / "labels" / "train" / f"{stem}.txt").read_text(encoding="utf-8").splitlines():
                cells = line.split()
                if len(cells) != 5 or cells[0] not in {"0", "1"}:
                    raise ValueError("invalid YOLO label")
                xywh = [float(v) for v in cells[1:]]
                if any(not 0 <= v <= 1 for v in xywh) or min(xywh[2:]) <= 0:
                    raise ValueError("invalid YOLO coordinates")
                x, y, w, h = xywh
                if x - w / 2 < -1e-6 or y - h / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6:
                    raise ValueError("YOLO box extends beyond the image")
                boxes[int(cells[0])] += 1
        counts[name] = {"images": len(chosen), "scratch": boxes[0], "hole": boxes[1]}
        if not all(boxes):
            raise ValueError(f"{name} is missing a defect class")
    out.mkdir(parents=True)
    for name, paths in manifests.items():
        (out / f"{name}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    cfg = {"path": dataset.as_posix(), "train": (out / "train.txt").as_posix(),
           "val": (out / "val.txt").as_posix(), "names": {0: "scratch", 1: "hole"}}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"fold": held_out_fold, "counts": counts,
        "warning": "Grouping is approximate; physical cable segments have not been verified."}, indent=2) + "\n",
        encoding="utf-8")
    return out / "data.yaml", counts
