#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/analyze.py
====================
Produces a consolidated comparison table for ALL models:

  REVE models (reve_pipeline results):
    - REVE Baseline   → results/A-C/baseline_results/
    - REVE End2End    → results/A-C/end2end_phase2_*/   (latest run)

  Foundation model benchmarks (frozen + SVM):
    - LEAD            → benchmark/lead/results/
    - LaBraM          → benchmark/LaBraM/results/
    - BIOT            → benchmark/BIOT/results/

Outputs:
    results/summary_table.csv   — one row per model, all metrics
"""

import sys
import json
import csv
import glob
from pathlib import Path
from typing import Optional, Dict

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from reve_pipeline.common.paths import REPO_ROOT, RESULTS_ROOT

# ===========================================================================
# Model source registry
# ===========================================================================
#   Each entry: (display_name, results_path_or_None, strategy)
#   strategy = "fixed"        → results_path is the exact folder
#   strategy = "latest_glob"  → find latest subfolder matching glob
#              ("glob_pattern", parent_dir)

MODELS = [
    ("REVE Baseline",  RESULTS_ROOT / "A-C" / "baseline_results",  "fixed"),
    ("REVE End2End",   str(RESULTS_ROOT / "A-C"),                   "latest_glob"),
    ("LEAD",           REPO_ROOT / "benchmark" / "lead"  / "results", "fixed"),
    ("LaBraM",         REPO_ROOT / "benchmark" / "LaBraM" / "results", "fixed"),
    ("BIOT",           REPO_ROOT / "benchmark" / "BIOT"  / "results", "fixed"),
]

METRICS_KEYS = ["acc", "bal_acc", "mcc", "macro_f1", "weighted_f1", "sens", "spec"]


def _find_latest_end2end(parent_dir: str) -> Optional[Path]:
    """Return the most recent end2end_phase2_* subfolder."""
    runs = sorted(Path(parent_dir).glob("end2end_phase2_*"))
    return runs[-1] if runs else None


def _load_run_meta(results_dir: Path) -> Optional[Dict]:
    meta_path = results_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []

    for display_name, path_arg, strategy in MODELS:
        if strategy == "fixed":
            run_dir = Path(path_arg)
        elif strategy == "latest_glob":
            run_dir = _find_latest_end2end(path_arg)
            if run_dir is None:
                print(f"[WARN] No end2end run found for {display_name} — skipping.")
                continue
        else:
            print(f"[WARN] Unknown strategy '{strategy}' for {display_name}.")
            continue

        obj = _load_run_meta(run_dir)
        if obj is None:
            print(f"[WARN] run_meta.json not found in {run_dir} — skipping.")
            continue

        meta    = obj.get("meta", {})
        g       = obj.get("global_metrics", {})
        g_std   = obj.get("global_metrics_std", {})
        g_fmt   = obj.get("global_metrics_formatted", {})

        row = {
            "model":       display_name,
            "run_dir":     str(run_dir),
            "created_at":  meta.get("created_at", ""),
            "n_subjects":  meta.get("n_subjects", ""),
            "phase":       meta.get("phase", ""),
        }
        for k in METRICS_KEYS:
            row[k]            = g.get(k, "")
            row[f"{k}_std"]   = g_std.get(k, "")
            row[f"{k}_fmt"]   = g_fmt.get(k, "")

        rows.append(row)
        print(f"[OK] {display_name:20s}  acc={g.get('acc', '?'):.4f}  "
              f"mcc={g.get('mcc', '?'):.4f}")

    if not rows:
        print("No results found — nothing to write.")
        return

    out_csv = RESULTS_ROOT / "summary_table.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 70)
    print(f"Summary table → {out_csv}")
    print(f"  {len(rows)} model(s) reported.")
    print("=" * 70)


if __name__ == "__main__":
    main()
