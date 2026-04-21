#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LEAD Linear Probing Benchmark
================================
Uses LEAD P-Base as a frozen feature extractor.
LOSO cross-validation with SVM classifier (Linear Probing).

Shared LOSO/SVM/save logic lives in benchmark/common/benchmark_utils.py.
Only LEAD-specific model loading and embedding extraction is here.
"""

import sys
import random
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BENCHMARK_DIR  = Path(__file__).parent.parent          # benchmark/
_PROJECT_ROOT   = _BENCHMARK_DIR.parent                 # alz-ftd-ctl-reve/
_LEAD_REPO      = Path(__file__).parent / "repo"

for p in [str(_PROJECT_ROOT), str(_BENCHMARK_DIR), str(_LEAD_REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    import torch.nn as nn
    from models.LEADv2 import Model as LEADv2Model
except Exception as e:
    print(f"Dependency error loading LEAD: {e}")
    sys.exit(1)

from common.benchmark_utils import (
    load_subjects, run_loso_svm, compute_metrics_bootstrapped, save_results
)
from reve_pipeline.common.cache_io import load_window_pkl

# ===========================================================================
# CONFIG
# ===========================================================================
TARGET_LABELS = ["A", "C"]
SEED          = 42
RESULTS_DIR   = Path(__file__).parent / "results"

# LEAD P-Base architecture hyper-params
LEAD_CONFIG = dict(
    patch_size=50, stride=25, emb_dim=128, depth=8, num_heads=8,
    mlp_ratio=4.0, qkv_bias=True, norm_layer="layer", pool="cls",
    n_channels=19,
)


class _Cfg:
    """Minimal config-object that LEAD model expects."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ===========================================================================
# Model loading
# ===========================================================================

def load_frozen_lead(ckpt_path: Path, device: str):
    cfg   = _Cfg(**LEAD_CONFIG)
    model = LEADv2Model(cfg).to(device)

    print(f"Loading LEAD P-Base checkpoint from {ckpt_path} …")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    state_dict = ckpt.get("model", ckpt)
    cleaned    = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  Unexpected    ({len(unexpected)}): {unexpected[:3]}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("LEAD P-Base loaded and frozen.")
    return model


# ===========================================================================
# Feature extraction
# ===========================================================================

def embed_subject(model, X_win: np.ndarray, device: str) -> np.ndarray:
    """(N, C, T) → (embed_dim,) subject-level CLS embedding."""
    model.eval()
    batch_size = 128
    embs = []
    with torch.no_grad():
        for s in range(0, X_win.shape[0], batch_size):
            batch = torch.from_numpy(
                X_win[s:s + batch_size].astype(np.float32)
            ).to(device)
            out = model(batch)           # (B, embed_dim)
            if out.dim() == 3:
                out = out[:, 0, :]       # CLS token
            embs.append(out.cpu().numpy())
    all_emb = np.concatenate(embs, axis=0)
    return np.mean(all_emb, axis=0)


# ===========================================================================
# Main
# ===========================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"LEAD Linear Probing Benchmark | Device: {device}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ckpt_path = Path(__file__).parent / "checkpoints" / "checkpoint.pth"
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    model    = load_frozen_lead(ckpt_path, device)
    subjects, id_to_label = load_subjects(TARGET_LABELS)
    n_classes = len(id_to_label)
    print(f"Found {len(subjects)} subjects matching {TARGET_LABELS}.")

    # --- feature extraction ---
    print("Extracting embeddings using frozen LEAD P-Base …")
    all_embeddings, all_labels = [], []
    for subj in subjects:
        _, X_win, _ = load_window_pkl(subj["pkl"])
        h = embed_subject(model, X_win, device)
        all_embeddings.append(h)
        all_labels.append(subj["label"])
        print(f"  Extracted {subj['sid']}  shape={h.shape}")

    X = np.array(all_embeddings)
    y = np.array(all_labels)
    print(f"All embeddings shape: {X.shape}")

    # --- LOSO SVM ---
    print("Running LOSO SVM evaluation …")
    y_true, y_pred, fold_rows = run_loso_svm(X, y, subjects, id_to_label, SEED)

    # --- metrics ---
    metrics, metrics_std, formatted = compute_metrics_bootstrapped(
        y_true, y_pred, n_classes, SEED
    )

    # --- save ---
    meta = {
        "meta": {
            "run_id":        "LEAD_linear_probe_benchmark",
            "created_at":    datetime.now().isoformat(),
            "phase":         "linear_probe_benchmark",
            "model":         "LEAD P-Base",
            "embed_dim":     LEAD_CONFIG["emb_dim"],
            "target_labels": TARGET_LABELS,
            "n_subjects":    len(subjects),
        },
        "global_metrics":           metrics,
        "global_metrics_std":       metrics_std,
        "global_metrics_formatted": formatted,
    }
    save_results(RESULTS_DIR, fold_rows, y_true, y_pred, n_classes, id_to_label, meta)

    print("=" * 78)
    print("LEAD Linear Probing Benchmark DONE →", str(RESULTS_DIR))
    for k, v in formatted.items():
        print(f"  {k}: {v}")
    print("=" * 78)


if __name__ == "__main__":
    main()
