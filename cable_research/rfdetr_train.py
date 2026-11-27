"""Fine-tune RF-DETR Nano with a local checkpoint without network access."""

import argparse
import os
import socket
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True, help="official local RF-DETR Nano checkpoint")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accumulate", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if not 1 <= a.epochs <= 300 or a.resolution % 32 or a.batch < 1 or a.accumulate < 1:
        ap.error("invalid training budget or Nano resolution")
    dataset, weights, work = a.dataset_dir.resolve(strict=True), a.weights.resolve(strict=True), a.work.resolve()
    if not (dataset / "data.yaml").is_file() or work.exists():
        raise ValueError("missing dataset YAML or output directory already exists")
    work.mkdir(parents=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ.pop("ROBOFLOW_API_KEY", None)

    def offline(*args, **kwargs):
        raise OSError("network disabled during private-data training")

    socket.create_connection = offline
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    import torch
    from rfdetr import RFDETRNano

    torch.set_num_threads(4)
    model = RFDETRNano(pretrain_weights=str(weights), resolution=a.resolution)
    model.train(dataset_dir=str(dataset), output_dir=str(work), epochs=a.epochs,
                batch_size=a.batch, grad_accum_steps=a.accumulate, lr=1e-4,
                num_workers=0, seed=a.seed, tensorboard=False, wandb=False,
                early_stopping=False, use_ema=True)
    print("Completed local RF-DETR Nano experiment", flush=True)


if __name__ == "__main__":
    main()
