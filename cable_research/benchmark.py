"""Measure a trusted local checkpoint's network-only FP32 latency."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def measure(model, x, warmup=20, trials=100):
    if warmup < 0 or trials < 2:
        raise ValueError("invalid warmup or trial count")
    model.eval()
    device = x.device
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = []
        for _ in range(trials):
            if device.type == "cuda":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                end.synchronize()
                times.append(float(start.elapsed_time(end)))
            else:
                start = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - start) * 1000)
    ordered = sorted(times)
    return {"median_ms": statistics.median(ordered), "mean_ms": statistics.mean(ordered),
            "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="trusted local checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="new JSON output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--family", choices=("yolo", "rtdetr"), default="yolo")
    a = parser.parse_args()
    if a.out.exists() or a.imgsz < 128:
        raise ValueError("invalid output or image size")
    from ultralytics import RTDETR, YOLO

    # Import the checkpoint's custom block before loading trusted local weights.
    from cable_research import mixer  # noqa: F401

    checkpoint = a.weights.resolve(strict=True)
    model = (YOLO(checkpoint) if a.family == "yolo" else RTDETR(checkpoint)).model.float().to(a.device)
    device = torch.device(a.device)
    generator = torch.Generator().manual_seed(123)
    x = torch.rand((1, 3, a.imgsz, a.imgsz), generator=generator).to(device)
    result = {"network_only_fp32": measure(model, x, a.warmup, a.trials),
              "parameters": sum(p.numel() for p in model.parameters()),
              "input": [1, 3, a.imgsz, a.imgsz], "device": str(device), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "excluded": "preprocessing, postprocessing, disk I/O, export backend and transfer"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
