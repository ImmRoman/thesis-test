"""
scripts/run_experiments.py

Systematic experiment runner for the thesis.
Trains and evaluates all model × dataset combinations,
then saves a summary CSV for easy comparison.

Usage:
    python scripts/run_experiments.py --plan configs/experiment_plan.yaml
    python scripts/run_experiments.py --dry_run   ← just print what would run
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_PLAN = {
    "experiments": [
        # ── YOLOv8 ──
        {"name": "yolov8l_bdd100k",  "config": "configs/yolo/yolov8_bdd100k.yaml",
         "overrides": ["model.variant=yolov8l"]},
        {"name": "yolov8x_bdd100k",  "config": "configs/yolo/yolov8_bdd100k.yaml",
         "overrides": ["model.variant=yolov8x"]},
        # ── YOLOv10 (NMS-free) ──
        {"name": "yolov10l_bdd100k", "config": "configs/yolo/yolov8_bdd100k.yaml",
         "overrides": ["model.variant=yolov10l", "logging.run_name=yolov10l_bdd100k"]},
        # ── RT-DETR ──
        {"name": "rtdetr_l_bdd100k", "config": "configs/rtdetr/rtdetr_kitti.yaml",
         "overrides": ["data.dataset=bdd100k", "logging.run_name=rtdetr_l_bdd100k"]},
        {"name": "rtdetr_x_bdd100k", "config": "configs/rtdetr/rtdetr_kitti.yaml",
         "overrides": ["model.variant=rtdetr-x", "data.dataset=bdd100k",
                       "logging.run_name=rtdetr_x_bdd100k"]},
    ],
    # After training, evaluate each model also on these datasets
    "eval_datasets": ["bdd100k", "kitti"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan",    type=str, default=None,
                        help="YAML experiment plan (uses built-in default if omitted)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output",  default="runs/experiment_results.csv")
    return parser.parse_args()


def run_cmd(cmd: list, dry_run: bool = False) -> int:
    print(f"\n$ {' '.join(cmd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    return result.returncode


def find_best_checkpoint(run_name: str, output_dir: str = "runs") -> str | None:
    candidates = list(Path(output_dir).glob(f"{run_name}/**/best.pt"))
    if candidates:
        return str(candidates[0])
    candidates = list(Path(output_dir).glob(f"{run_name}/**/*.pt"))
    return str(candidates[0]) if candidates else None


def main():
    args = parse_args()

    if args.plan:
        with open(args.plan) as f:
            plan = yaml.safe_load(f)
    else:
        plan = DEFAULT_PLAN

    experiments = plan["experiments"]
    eval_datasets = plan.get("eval_datasets", ["bdd100k", "kitti"])

    print(f"{'='*60}")
    print(f"  Experiment Plan — {len(experiments)} runs")
    print(f"  Eval datasets: {eval_datasets}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}")

    results = []

    for exp in experiments:
        name    = exp["name"]
        config  = exp["config"]
        ovrd    = exp.get("overrides", [])

        print(f"\n{'─'*60}")
        print(f"  TRAINING: {name}")
        print(f"{'─'*60}")
        t0 = time.time()

        rc = run_cmd(
            ["python", "scripts/train.py", "--config", config] + ovrd,
            dry_run=args.dry_run,
        )
        train_time = time.time() - t0

        if rc != 0:
            print(f"  [WARNING] Training returned code {rc}")

        # ── Evaluation on each dataset ───────────────────────────────────
        for ds in eval_datasets:
            ckpt = find_best_checkpoint(name)
            if ckpt is None and not args.dry_run:
                print(f"  [WARNING] No checkpoint found for {name}, skipping eval")
                continue

            ckpt = ckpt or f"runs/{name}/weights/best.pt"
            eval_out = f"runs/{name}/eval_{ds}.json"

            rc_eval = run_cmd(
                [
                    "python", "scripts/eval.py",
                    "--config", config,
                    "--checkpoint", ckpt,
                    "--dataset", ds,
                    "--split", "test",
                ] + ovrd,
                dry_run=args.dry_run,
            )

            # Load results
            metrics = {}
            if not args.dry_run and Path(eval_out).exists():
                with open(eval_out) as f:
                    metrics = json.load(f)

            results.append({
                "experiment": name,
                "eval_dataset": ds,
                "train_time_s": round(train_time, 1),
                "mAP50":    metrics.get("mAP50",    None),
                "mAP50-95": metrics.get("mAP50-95", None),
                "timestamp": datetime.now().isoformat(timespec="minutes"),
            })

    # ── Summary ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
