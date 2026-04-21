#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/plot_grouped_bar.py
=============================
Grouped bar chart comparing ALL models across key metrics.

Reads run_meta.json from:
  - REVE Baseline   → results/A-C/baseline_results/
  - REVE End2End    → results/A-C/end2end_phase2_*/  (latest)
  - LEAD            → benchmark/lead/results/
  - LaBraM          → benchmark/LaBraM/results/
  - BIOT            → benchmark/BIOT/results/

Output: results/grouped_metrics_barplot.png
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from reve_pipeline.common.paths import REPO_ROOT, RESULTS_ROOT

# ---------------------------------------------------------------------------
# Registry (same order as analyze.py)
# ---------------------------------------------------------------------------
MODELS = [
    ("REVE Baseline",  RESULTS_ROOT / "A-C" / "baseline_results",         "fixed"),
    ("REVE + C2C",   RESULTS_ROOT / "A-C",                               "latest_end2end"),
    ("LEAD",           REPO_ROOT / "benchmark" / "lead"   / "results",     "fixed"),
    ("LaBraM",         REPO_ROOT / "benchmark" / "LaBraM" / "results",     "fixed"),
    ("BIOT",           REPO_ROOT / "benchmark" / "BIOT"   / "results",     "fixed"),
]

METRICS       = ["acc", "mcc", "macro_f1", "sens", "spec"]
METRIC_LABELS = ["Accuracy", "MCC", "Macro F1","Specificity", "Sensitivity"]
OUTPUT_PLOT   = RESULTS_ROOT / "grouped_metrics_barplot.png"


def _find_latest_end2end(parent: Path):
    runs = sorted(parent.glob("end2end_phase2_*"))
    return runs[-1] if runs else None


def _load(run_dir: Path):
    p = run_dir / "run_meta.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    run_data = []
    for display_name, path_arg, strategy in MODELS:
        if strategy == "fixed":
            run_dir = Path(path_arg)
        elif strategy == "latest_end2end":
            run_dir = _find_latest_end2end(Path(path_arg))
            if run_dir is None:
                print(f"[SKIP] No end2end run for {display_name}")
                continue
        else:
            continue

        obj = _load(run_dir)
        if obj is None:
            print(f"[SKIP] run_meta.json missing: {run_dir}")
            continue

        g     = obj.get("global_metrics", {})
        g_std = obj.get("global_metrics_std", {})
        if g:
            run_data.append({
                "name":    display_name,
                "metrics": g,
                "std":     g_std,
            })

    if not run_data:
        print("No data found — nothing to plot.")
        return

    # -----------------------------------------------------------------------
    # Draw
    # -----------------------------------------------------------------------
    x     = np.arange(len(METRICS))
    width = 0.8 / len(run_data)
    cmap  = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(15, 8))

    for i, run in enumerate(run_data):
        means  = [run["metrics"].get(m, 0) for m in METRICS]
        stds   = [run["std"].get(m, 0)     for m in METRICS]
        offset = (i - len(run_data) / 2) * width + width / 2

        rects = ax.bar(
            x + offset, means, width,
            label=run["name"], yerr=stds, capsize=4,
            color=cmap(i % 10), alpha=0.9, edgecolor="black",
        )
        for j, rect in enumerate(rects):
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h + stds[j]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", rotation=90,
                    fontsize=9, fontweight="bold",
                )

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Comparison (Alzheimer's vs Healthy Controls)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(title="Models", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)

    fig.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
