"""Materialize a private YOLO directory layout for external detector libraries."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from cable_research.split import create_split


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.digest()


def materialize(dataset: Path, fold_plan: Path, out: Path, fold: int):
    dataset, out = dataset.resolve(strict=True), out.resolve()
    if out.exists():
        raise FileExistsError("refusing to overwrite external dataset")
    _, counts = create_split(dataset, fold_plan, out / "_split", fold)
    for split, target in (("train", "train"), ("val", "valid")):
        image_dir, label_dir = out / target / "images", out / target / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        images = (out / "_split" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        for line in images:
            source_image = Path(line)
            source_label = dataset / "labels/train" / f"{source_image.stem}.txt"
            dest_image, dest_label = image_dir / source_image.name, label_dir / source_label.name
            shutil.copyfile(source_image, dest_image)
            shutil.copyfile(source_label, dest_label)
            if digest(source_image) != digest(dest_image) or digest(source_label) != digest(dest_label):
                raise OSError("external dataset copy verification failed")
    cfg = {"train": "train/images", "val": "valid/images", "nc": 2, "names": {0: "scratch", 1: "hole"}}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "copy_receipt.json").write_text(json.dumps({"fold": fold, "counts": counts,
        "image_and_label_copies_sha256_verified": True,
        "warning": "Physical cable group membership has not been independently verified."}, indent=2) + "\n",
        encoding="utf-8")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--fold-plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fold", type=int, choices=(1, 2, 3), default=2)
    a = ap.parse_args()
    print("Prepared private external-detector data:", materialize(a.dataset, a.fold_plan, a.out, a.fold))


if __name__ == "__main__":
    main()
