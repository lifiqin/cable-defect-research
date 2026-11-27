"""Plan or execute an explicit local, bounded factorial ablation grid."""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

MODES = {"baseline", "isotropic", "uniform", "tensor", "contrast", "hybrid"}


def combinations(modes, sites, gates, seeds):
    if (not modes or not sites or not gates or not seeds or not set(modes) <= MODES or
            not set(sites) <= {"p3", "p3p4"} or not set(gates) <= {0, 1} or
            any(not isinstance(seed, int) or seed < 0 for seed in seeds)):
        raise ValueError("invalid or empty grid axis")
    if any(len(axis) != len(set(axis)) for axis in (modes, sites, gates, seeds)):
        raise ValueError("duplicate values on grid axes")
    rows = []
    for mode in modes:
        choices = itertools.product(["p3"], [0], seeds) if mode == "baseline" else itertools.product(sites, gates, seeds)
        for site, gate, seed in choices:
            name = f"{mode}_{site}_gate{gate}_seed{seed}"
            rows.append({"name": name, "mode": mode, "sites": site, "gate": bool(gate), "seed": seed})
    if not rows or len(rows) > 24:
        raise ValueError("grid must contain 1 to 24 runs")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--fold-plan", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--modes", nargs="+", choices=sorted(MODES), default=["baseline", "isotropic", "tensor", "hybrid"])
    ap.add_argument("--sites", nargs="+", choices=("p3", "p3p4"), default=["p3"])
    ap.add_argument("--gates", nargs="+", type=int, choices=(0, 1), default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--fold", type=int, choices=(1, 2, 3), default=2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--cache", choices=("ram", "none"), default="ram")
    ap.add_argument("--optimizer", choices=("AdamW", "SGD", "auto"), default="AdamW")
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--execute", action="store_true", help="actually run all jobs; otherwise only print the plan")
    a = ap.parse_args()
    specs = combinations(a.modes, a.sites, a.gates, a.seeds)
    print(json.dumps(specs, indent=2))
    print(f"Total {len(specs)} jobs; repeated validation selection is not independent testing.", flush=True)
    if not a.execute:
        return
    for source in (a.dataset, a.fold_plan, a.weights):
        source.resolve(strict=True)
    work = a.work.resolve()
    if work.exists():
        raise FileExistsError("refusing to reuse an existing grid directory")
    work.mkdir(parents=True)
    (work / "grid_plan.json").write_text(json.dumps(specs, indent=2) + "\n", encoding="utf-8")
    for spec in specs:
        name = spec["name"]
        command = [sys.executable, "-m", "cable_research.train", "--dataset", str(a.dataset.resolve()),
                   "--fold-plan", str(a.fold_plan.resolve()), "--weights", str(a.weights.resolve()),
                   "--work", str(work / name), "--mode", spec["mode"], "--sites", spec["sites"],
                   "--seed", str(spec["seed"]), "--fold", str(a.fold), "--epochs", str(a.epochs),
                   "--imgsz", str(a.imgsz), "--batch", str(a.batch), "--device", a.device, "--cache", a.cache,
                   "--optimizer", a.optimizer, "--lr0", str(a.lr0)]
        if spec["gate"]:
            command.append("--channel-gate")
        print("Starting", name, flush=True)
        with (work / f"{name}.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("Completed", name, flush=True)


if __name__ == "__main__":
    main()
