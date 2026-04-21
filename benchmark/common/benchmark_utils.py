#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark/common/benchmark_utils.py
====================================
Shared utilities for all Linear Probing benchmarks (LEAD, LaBraM, BIOT, …).

Eliminates ~200 lines of duplicated code from each benchmark script by
providing:
  - load_subjects()               — load subjects from window cache
  - run_loso_svm()                — LOSO cross-validation with SVC
  - compute_metrics_bootstrapped()— metrics + bootstrap std (1000 iterations)
  - save_results()                — fold CSV, CM, CM plot, run_meta.json
"""

import os
import sys
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so reve_pipeline can be imported from
# any working directory.
# benchmark/common/benchmark_utils.py → parent=common → parent=benchmark
#                                      → parent=project_root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from reve_pipeline.common.cache_io import list_pkls, load_window_pkl
from reve_pipeline.common.labels import load_labels
from reve_pipeline.common.metrics import compute_metrics, compute_confusion_matrix
from reve_pipeline.common.paths import WINDOW_DIR, PARTICIPANTS_TXT


# ===========================================================================
# Subject loading
# ===========================================================================

def load_subjects(target_labels):
    """
    Load subjects from the window cache that match *target_labels*.

    Returns
    -------
    subjects : list of dict {sid, pkl, label}
        Sorted by subject ID.
    id_to_label : dict {int → str}
        Mapping from integer class index to label string.
    """
    label_dict, _label_map, id_to_label = load_labels(
        str(PARTICIPANTS_TXT), target_labels=target_labels
    )
    pkls = list_pkls(WINDOW_DIR)
    subjects = []
    for p in pkls:
        sid = os.path.basename(p).split("_windows.pkl")[0]
        if sid in label_dict:
            subjects.append({"sid": sid, "pkl": p, "label": int(label_dict[sid])})
    subjects = sorted(subjects, key=lambda x: x["sid"])
    return subjects, id_to_label


# ===========================================================================
# LOSO SVM
# ===========================================================================

def run_loso_svm(X, y, subjects, id_to_label, seed):
    """
    Leave-One-Subject-Out cross-validation using sklearn SVC.

    Parameters
    ----------
    X : np.ndarray, shape (n_subjects, embed_dim)
    y : np.ndarray, shape (n_subjects,)
    subjects : list of dict produced by load_subjects()
    id_to_label : dict {int → str}
    seed : int

    Returns
    -------
    y_true_all : np.ndarray
    y_pred_all : np.ndarray
    fold_rows  : list of dict (one per fold, ready for CSV)
    """
    n_subjects = len(subjects)
    y_true_all, y_pred_all, fold_rows = [], [], []

    for i in range(n_subjects):
        test_subj = subjects[i]["sid"]
        test_y    = y[i]

        X_test  = X[i].reshape(1, -1)
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)

        clf = make_pipeline(
            StandardScaler(),
            SVC(probability=True, random_state=seed)
        )
        t0   = time.time()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]
        prob = clf.predict_proba(X_test)[0]
        dt   = time.time() - t0

        y_true_all.append(int(test_y))
        y_pred_all.append(int(pred))

        row = {
            "fold_idx":      i + 1,
            "test_subject":  test_subj,
            "y_true":        int(test_y),
            "y_pred":        int(pred),
            "y_true_label":  id_to_label[int(test_y)],
            "y_pred_label":  id_to_label[int(pred)],
            "runtime_sec":   dt,
        }
        for cls_idx in range(len(prob)):
            row[f"prob_{id_to_label[cls_idx]}"] = prob[cls_idx]
        fold_rows.append(row)

        print(
            f"  [Fold {i + 1}/{n_subjects}] test={test_subj} "
            f"true={id_to_label[int(test_y)]} pred={id_to_label[int(pred)]}"
        )

    return (
        np.array(y_true_all),
        np.array(y_pred_all),
        fold_rows,
    )


# ===========================================================================
# Metrics with bootstrap std
# ===========================================================================

def compute_metrics_bootstrapped(y_true, y_pred, n_classes, seed,
                                  n_iterations=1000):
    """
    Compute classification metrics and their bootstrap standard deviations.

    Returns
    -------
    metrics           : dict {metric_name → float}
    metrics_std       : dict {metric_name → float}
    formatted_metrics : dict {metric_name → "mean ± std" string}
    """
    metrics = compute_metrics(y_true, y_pred, n_classes=n_classes)

    n_samples    = len(y_true)
    rng_bs       = np.random.default_rng(seed + 555)
    metrics_list = []
    for _ in range(n_iterations):
        idx = rng_bs.choice(n_samples, n_samples, replace=True)
        m   = compute_metrics(y_true[idx], y_pred[idx], n_classes)
        metrics_list.append(m)

    metrics_std       = {}
    formatted_metrics = {}
    for k in metrics:
        vals            = [m[k] for m in metrics_list]
        std_val         = float(np.std(vals))
        metrics_std[k]  = std_val
        formatted_metrics[k] = f"{metrics[k]:.4f} \u00b1 {std_val:.4f}"

    return metrics, metrics_std, formatted_metrics


# ===========================================================================
# Saving results
# ===========================================================================

def save_results(results_dir, fold_rows, y_true, y_pred,
                 n_classes, id_to_label, meta_dict):
    """
    Persist:
      fold_results.csv
      confusion_matrix.npy / .csv / .png
      run_meta.json

    Parameters
    ----------
    results_dir : str | Path
    fold_rows   : list of dict (from run_loso_svm)
    y_true, y_pred : np.ndarray
    n_classes   : int
    id_to_label : dict {int → str}
    meta_dict   : dict written verbatim to run_meta.json
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- fold CSV ---
    fold_csv = results_dir / "fold_results.csv"
    with open(fold_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    # --- confusion matrix ---
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=n_classes)
    np.save(results_dir / "confusion_matrix.npy", cm)

    cm_csv = results_dir / "confusion_matrix.csv"
    with open(cm_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow([""] + [id_to_label[i] for i in range(n_classes)])
        for i in range(n_classes):
            wri.writerow([id_to_label[i]] + list(cm[i].tolist()))

    # --- confusion matrix plot ---
    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[id_to_label[i] for i in range(n_classes)],
        yticklabels=[id_to_label[i] for i in range(n_classes)],
        cbar=True, annot_kws={"size": 16},
    )
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    ax.tick_params(left=False, bottom=False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0, va="center")
    cbar = ax.collections[0].colorbar
    if hasattr(cbar, "ax"):
        cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # --- run_meta.json ---
    with open(results_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    print(f"Results saved → {results_dir}")
