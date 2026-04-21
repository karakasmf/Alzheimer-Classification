#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/statistical_test_all.py
==================================
Permutation Test (10 000 iterations) across all metric keys:
  REVE Baseline vs REVE End2End.

Reads from:
  - results/A-C/baseline_results/fold_results.csv
  - results/A-C/end2end_phase2_*/fold_results.csv  (latest run)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from reve_pipeline.common.paths import RESULTS_ROOT
from reve_pipeline.common.metrics import compute_metrics

N_PERMUTATIONS = 10_000
np.random.seed(42)


def _find_latest_end2end(parent: Path):
    runs = sorted(parent.glob("end2end_phase2_*"))
    return runs[-1] if runs else None


def main():
    ac_dir = RESULTS_ROOT / "A-C"
    baseline_dir = ac_dir / "baseline_results"
    proposed_dir = _find_latest_end2end(ac_dir)

    if proposed_dir is None:
        print("End2End klasörü bulunamadı.")
        return

    df_b = pd.read_csv(baseline_dir / "fold_results.csv")
    df_p = pd.read_csv(proposed_dir / "fold_results.csv")

    df_b = df_b.sort_values("test_subject").reset_index(drop=True)
    df_p = df_p.sort_values("test_subject").reset_index(drop=True)

    y_true   = df_b["y_true"].values
    y_pred_b = df_b["y_pred"].values
    y_pred_p = df_p["y_pred"].values

    n_classes   = len(np.unique(y_true))
    metrics_b   = compute_metrics(y_true, y_pred_b, n_classes)
    metrics_p   = compute_metrics(y_true, y_pred_p, n_classes)

    metric_keys = ["acc", "bal_acc", "mcc", "macro_f1", "sens", "spec"]
    delta_orig  = {k: abs(metrics_p.get(k, 0) - metrics_b.get(k, 0)) for k in metric_keys}
    count_exceed = {k: 0 for k in metric_keys}

    print(f"Permütasyon testi ({N_PERMUTATIONS:,} iterasyon) …")
    for _ in tqdm(range(N_PERMUTATIONS)):
        swap_idx  = np.random.rand(len(y_true)) > 0.5
        pseudo_b  = np.where(swap_idx, y_pred_p, y_pred_b)
        pseudo_p  = np.where(swap_idx, y_pred_b, y_pred_p)
        pm_b = compute_metrics(y_true, pseudo_b, n_classes)
        pm_p = compute_metrics(y_true, pseudo_p, n_classes)
        for k in metric_keys:
            if abs(pm_p.get(k, 0) - pm_b.get(k, 0)) >= delta_orig[k]:
                count_exceed[k] += 1

    p_values = {k: count_exceed[k] / N_PERMUTATIONS for k in metric_keys}

    print("\n" + "=" * 70)
    print("TÜM METRİKLER İÇİN İSTATİSTİKSEL KARŞILAŞTIRMA (Permutation Test)")
    print("=" * 70)
    print(f"{'Metrik':<15} | {'Baseline':<10} | {'Proposed':<10} | {'P-Value':<10} | Anlamlı?")
    print("-" * 70)
    for k in metric_keys:
        sig = "EVET" if p_values[k] < 0.05 else "HAYIR"
        print(f"{k:<15} | {metrics_b.get(k, 0):.4f}     | "
              f"{metrics_p.get(k, 0):.4f}     | {p_values[k]:.4f}     | {sig}")
    print("=" * 70)


if __name__ == "__main__":
    main()
