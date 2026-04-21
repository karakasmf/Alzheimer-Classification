#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/statistical_test.py
==============================
Paired T-Test: REVE Baseline vs REVE End2End.
Uses fold-level predictions from fold_results.csv.

Reads from:
  - results/A-C/baseline_results/fold_results.csv
  - results/A-C/end2end_phase2_*/fold_results.csv  (latest run)
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from reve_pipeline.common.paths import RESULTS_ROOT


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

    print(f"Baseline : {baseline_dir}")
    print(f"Proposed : {proposed_dir}")

    df_b = pd.read_csv(baseline_dir / "fold_results.csv")
    df_p = pd.read_csv(proposed_dir / "fold_results.csv")

    df_b = df_b.sort_values("test_subject").reset_index(drop=True)
    df_p = df_p.sort_values("test_subject").reset_index(drop=True)

    b_correct = (df_b["y_true"] == df_b["y_pred"]).astype(int)
    p_correct = (df_p["y_true"] == df_p["y_pred"]).astype(int)

    t_stat, p_val = stats.ttest_rel(b_correct, p_correct)

    print("-" * 50)
    print("PAIRED T-TEST SONUCLARI")
    print(f"T-Statistic : {t_stat:.4f}")
    print(f"P-Value     : {p_val:.5f}")
    if p_val < 0.05:
        print("→ İki model arasında istatistiksel olarak ANLAMLI bir fark var (p < 0.05).")
    else:
        print("→ İki model arasında istatistiksel olarak ANLAMLI BİR FARK YOK (p >= 0.05).")
    print("-" * 50)
    print(f"Baseline Doğru Sayısı : {b_correct.sum()} / {len(b_correct)} (Acc: {b_correct.mean():.4f})")
    print(f"Proposed Doğru Sayısı : {p_correct.sum()} / {len(p_correct)} (Acc: {p_correct.mean():.4f})")


if __name__ == "__main__":
    main()
