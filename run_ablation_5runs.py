"""
============================================================
Ablation Study: 5-Run Training + Per-Epoch Average Comparison
============================================================
Trains ALL five methods 5 times each on SSDD, then computes
per-epoch metric averages across runs for paper comparison.

Methods:
  M1 – YOLOv8n baseline           (train_yolov8n_baseline.py)
  M2 – YOLOv8n SAR Lightweight    (train_yolov8n_sar_lighweight.py)
  A1 – YOLOv8n + EDFS only        (train_yolov8n_sar_edfs_only.py)
  A2 – YOLOv8n + MSR-CBAM only    (train_yolov8n_sar_msrcbam_only.py)
  A3 – YOLOv10n baseline           (train_yolov10n.py)

Author: PhD Research
Date: 2026
============================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import glob
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

# ============================================================
# Configuration
# ============================================================
NUM_RUNS     = 5
DEFAULT_SEEDS = [0, 11, 22, 33, 44]       # one seed per run (override via --seeds)
WORKSPACE    = Path(__file__).parent
DATA_YAML    = WORKSPACE / "ssdd" / "datasets" / "data.yaml"

# Metrics to average
METRICS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]

# Loss columns (also averaged)
LOSS_COLS = [
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss",   "val/cls_loss",   "val/dfl_loss",
]

ALL_COLS = LOSS_COLS + METRICS

# Where per-run results are stored (project/name_runN)
METHODS = {
    "yolov8n_baseline": {
        "script": WORKSPACE / "train_yolov8n_baseline.py",
        "project": "runs/ssdd/yolov8n_ssdd_baseline",
        "name_prefix": "exp1_ssdd",
    },
    "yolov8n_sar_lightweight": {
        "script": WORKSPACE / "train_yolov8n_sar_lighweight.py",
        "project": "runs/ssdd/yolov8n_cbam_edge",
        "name_prefix": "lightweight_v9_multiscale_edgedir",
    },
    "yolov8n_edfs_only": {
        "script": WORKSPACE / "train_yolov8n_sar_edfs_only.py",
        "project": "runs/ssdd/yolov8n_cbam_edge",
        "name_prefix": "ablation_edfs_only",
    },
    "yolov8n_msrcbam_only": {
        "script": WORKSPACE / "train_yolov8n_sar_msrcbam_only.py",
        "project": "runs/ssdd/yolov8n_cbam_edge",
        "name_prefix": "ablation_msrcbam_only",
    },
    "yolov10n_baseline": {
        "script": WORKSPACE / "train_yolov10n.py",
        "project": "runs/ssdd/yolov10n_ssdd_baseline",
        "name_prefix": "exp1_ssdd",
    },
}

# Output folder for averaged CSVs
OUTPUT_DIR = WORKSPACE / "runs" / "ssdd" / "ablation_averages"

# ============================================================
# Helpers – patch a training script for a specific run
# ============================================================

def _patch_and_run(script_path: Path, project: str, run_name: str, seed: int):
    """
    Read the original script, patch seed/project/name, write a temp
    version, run it as a subprocess, then delete the temp file.
    """
    src = script_path.read_text(encoding="utf-8")

    # Patch seed – handles both literal integers (seed=0) and variable (seed=SEED)
    import re
    src = re.sub(r'seed\s*=\s*[\w]+', f'seed={seed}', src)

    # Patch project
    src = re.sub(
        r'project\s*=\s*["\'].*?["\']',
        f'project=r"{project}"',
        src,
    )

    # Patch name – handles plain strings and f-strings (name=f"...")
    src = re.sub(
        r'name\s*=\s*f?["\'].*?["\']',
        f'name="{run_name}"',
        src,
    )

    tmp = script_path.with_name(f"_tmp_{run_name}_{script_path.name}")
    tmp.write_text(src, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  RUNNING: {run_name}  (seed={seed})")
    print(f"  Script:  {tmp.name}")
    print(f"{'='*60}\n")

    try:
        subprocess.check_call(
            [sys.executable, str(tmp)],
            cwd=str(WORKSPACE),
        )
    finally:
        tmp.unlink(missing_ok=True)


# ============================================================
# Phase 1 – Train all methods x NUM_RUNS
# ============================================================

def train_all(seeds: list[int]):
    for method_key, cfg in METHODS.items():
        for run_idx in range(NUM_RUNS):
            seed = seeds[run_idx]
            run_name = f"{cfg['name_prefix']}_run{run_idx}"
            run_dir = WORKSPACE / cfg["project"] / run_name

            # Skip if results.csv already exists (resume-friendly)
            existing = list(run_dir.glob("results.csv")) if run_dir.exists() else []
            if existing:
                print(f"[SKIP] {method_key} run{run_idx} – results.csv already exists at {run_dir}")
                continue

            _patch_and_run(
                script_path=cfg["script"],
                project=cfg["project"],
                run_name=run_name,
                seed=seed,
            )

    print("\nAll training runs finished.")


# ============================================================
# Phase 2 – Compute per-epoch averages across 5 runs
# ============================================================

def _read_results_csv(path: Path) -> list[dict]:
    """Read a YOLO results.csv (may have leading spaces in headers)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {}
            for k, v in row.items():
                clean[k.strip()] = v.strip() if isinstance(v, str) else v
            rows.append(clean)
    return rows


def compute_averages():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []       # for the combined comparison CSV

    for method_key, cfg in METHODS.items():
        per_epoch: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {col: [] for col in ALL_COLS}
        )

        # Collect all run CSVs
        run_dirs = []
        for run_idx in range(NUM_RUNS):
            run_name = f"{cfg['name_prefix']}_run{run_idx}"
            run_dir = WORKSPACE / cfg["project"] / run_name
            csv_path = run_dir / "results.csv"
            if not csv_path.exists():
                print(f"[WARN] Missing: {csv_path}")
                continue
            run_dirs.append(csv_path)
            rows = _read_results_csv(csv_path)
            for row in rows:
                try:
                    epoch = int(float(row.get("epoch", "")))
                except (TypeError, ValueError):
                    continue
                for col in ALL_COLS:
                    raw = row.get(col, "")
                    try:
                        per_epoch[epoch][col].append(float(raw))
                    except (TypeError, ValueError):
                        pass

        n_runs = len(run_dirs)
        print(f"\n{method_key}: found {n_runs} run(s)")
        if n_runs == 0:
            continue

        # Write per-epoch average CSV for this method
        out_csv = OUTPUT_DIR / f"{method_key}_per_epoch_avg.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "n_runs"] + ALL_COLS)
            for epoch in sorted(per_epoch.keys()):
                vals = per_epoch[epoch]
                row = [epoch, n_runs]
                for col in ALL_COLS:
                    v = vals[col]
                    row.append(f"{mean(v):.6f}" if v else "")
                writer.writerow(row)
        print(f"  -> {out_csv}")

        # Keep the last-epoch average for the combined summary
        if per_epoch:
            last_epoch = max(per_epoch.keys())
            last_vals = per_epoch[last_epoch]
            entry = {"method": method_key, "n_runs": n_runs, "last_epoch": last_epoch}
            for col in ALL_COLS:
                v = last_vals[col]
                entry[col] = f"{mean(v):.6f}" if v else ""
            summary_rows.append(entry)

    # Write combined summary CSV
    if summary_rows:
        summary_csv = OUTPUT_DIR / "methods_comparison_summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "n_runs", "last_epoch"] + ALL_COLS)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary: {summary_csv}")

    # Print comparison table to console
    print("\n" + "=" * 90)
    print("  PER-METHOD COMPARISON (last-epoch averages across runs)")
    print("=" * 90)
    header = f"{'Method':<28} {'Runs':>4}  {'Prec':>8}  {'Recall':>8}  {'mAP50':>8}  {'mAP50-95':>9}"
    print(header)
    print("-" * 90)
    for entry in summary_rows:
        print(
            f"{entry['method']:<28} {entry['n_runs']:>4}  "
            f"{entry.get('metrics/precision(B)', 'n/a'):>8}  "
            f"{entry.get('metrics/recall(B)', 'n/a'):>8}  "
            f"{entry.get('metrics/mAP50(B)', 'n/a'):>8}  "
            f"{entry.get('metrics/mAP50-95(B)', 'n/a'):>9}"
        )
    print("=" * 90)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation: train 5 runs & average")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase and only compute averages from existing results",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        metavar="SEED",
        help=(
            f"Space-separated list of {NUM_RUNS} integer seeds, one per run "
            f"(default: {DEFAULT_SEEDS}). "
            "Example: --seeds 42 7 123 999 2024"
        ),
    )
    args = parser.parse_args()

    if len(args.seeds) != NUM_RUNS:
        parser.error(
            f"--seeds requires exactly {NUM_RUNS} values, got {len(args.seeds)}: {args.seeds}"
        )

    start = datetime.now()
    print(f"Started at {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {args.seeds}")
    print(f"Methods: {list(METHODS.keys())}")

    if not args.skip_training:
        train_all(args.seeds)

    compute_averages()

    elapsed = datetime.now() - start
    print(f"\nTotal time: {elapsed}")
