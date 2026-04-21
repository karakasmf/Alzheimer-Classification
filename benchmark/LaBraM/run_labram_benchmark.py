#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaBraM Linear Probing Benchmark
================================
Uses LaBraM-Base as a frozen feature extractor.
LOSO cross-validation with SVM classifier (Linear Probing).

Shared LOSO/SVM/save logic lives in benchmark/common/benchmark_utils.py.
Only LaBraM-specific model loading and embedding extraction is here.
"""

import sys
import random
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BENCHMARK_DIR = Path(__file__).parent.parent          # benchmark/
_PROJECT_ROOT  = _BENCHMARK_DIR.parent                 # alz-ftd-ctl-reve/
_LABRAM_REPO   = Path(__file__).parent / "repo"

for p in [str(_PROJECT_ROOT), str(_BENCHMARK_DIR), str(_LABRAM_REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    import torch.nn as nn
    import modeling_finetune  # noqa: F401
    from modeling_finetune import NeuralTransformer
except Exception as e:
    print(f"Dependency error loading LaBraM: {e}")
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
PATCH_SIZE    = 200    # samples per patch (200 Hz × 1 s)
RESULTS_DIR   = Path(__file__).parent / "results"

# Our 19 channels → must be UPPERCASE to match LaBraM's standard_1020 list.
OUR_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T7",  "C3",  "CZ", "C4", "T8",
    "P7",  "P3",  "PZ", "P4", "P8",
    "O1",  "O2",
]

# standard_1020 from LaBraM repo/utils.py (lines 42-57)
STANDARD_1020 = [
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9",  "F7",  "F5",  "F3",  "F1",  "FZ",  "F2",  "F4",  "F6",  "F8",  "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9",  "T7",  "C5",  "C3",  "C1",  "CZ",  "C2",  "C4",  "C6",  "T8",  "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9",  "P7",  "P5",  "P3",  "P1",  "PZ",  "P2",  "P4",  "P6",  "P8",  "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1",  "OZ",  "O2",  "O9",  "CB1", "CB2",
    "IZ",  "O10", "T3",  "T5",  "T4",  "T6",  "M1",  "M2",  "A1",  "A2",
    "CFC1","CFC2","CFC3","CFC4","CFC5","CFC6","CFC7","CFC8",
    "CCP1","CCP2","CCP3","CCP4","CCP5","CCP6","CCP7","CCP8",
    "T1",  "T2",  "FTT9h","TTP7h","TPP9h","FTT10h","TPP8h","TPP10h",
]


def build_input_chans(our_channels):
    """Map our channel names to LaBraM positional embedding indices."""
    input_chans = [0]   # 0 = CLS token
    missing = []
    for ch in our_channels:
        if ch in STANDARD_1020:
            input_chans.append(STANDARD_1020.index(ch) + 1)
        else:
            missing.append(ch)
    if missing:
        print(f"[WARNING] Channels not found in standard_1020, skipped: {missing}")
    return input_chans


# ===========================================================================
# Model loading
# ===========================================================================

def load_frozen_labram(ckpt_path: Path, device: str):
    model = NeuralTransformer(
        EEG_size=PATCH_SIZE, patch_size=PATCH_SIZE,
        in_chans=1, out_chans=8, num_classes=0,
        embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.0,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_abs_pos_emb=True, use_rel_pos_bias=False,
        use_mean_pooling=True, init_values=0,
    ).to(device)

    print(f"Loading LaBraM-Base checkpoint from {ckpt_path} …")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.module.", "").replace("module.", "")
        if new_k.startswith("student."):
            new_k = new_k[len("student."):]
        elif new_k.startswith("teacher."):
            continue
        if new_k == "logit_scale":
            continue
        cleaned[new_k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  Unexpected    ({len(unexpected)}): {unexpected[:3]}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("LaBraM-Base loaded and frozen.")
    return model


# ===========================================================================
# Feature extraction
# ===========================================================================

def embed_subject(model, X_win: np.ndarray, input_chans: list,
                  device: str) -> np.ndarray:
    """(N, C, T) → (200,) subject-level embedding."""
    model.eval()
    batch_size = 64
    n_windows, n_ch, n_samples = X_win.shape
    n_patches = n_samples // PATCH_SIZE   # 400 // 200 = 2
    embs = []

    with torch.no_grad():
        for s in range(0, n_windows, batch_size):
            batch = X_win[s:s + batch_size].astype(np.float32)
            B = batch.shape[0]
            x = torch.from_numpy(
                batch.reshape(B, n_ch, n_patches, PATCH_SIZE)
            ).to(device)
            emb = model.forward_features(x, input_chans=input_chans)  # (B, 200)
            embs.append(emb.cpu().numpy())

    all_emb = np.concatenate(embs, axis=0)   # (N, 200)
    return np.mean(all_emb, axis=0)          # (200,)


# ===========================================================================
# Main
# ===========================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"LaBraM Linear Probing Benchmark | Device: {device}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ckpt_path = Path(__file__).parent / "checkpoints" / "labram-base.pth"
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    input_chans = build_input_chans(OUR_CHANNELS)
    print(
        f"Channel mapping: {len(input_chans) - 1} channels mapped "
        f"(+1 CLS). input_chans={input_chans}"
    )

    model    = load_frozen_labram(ckpt_path, device)
    subjects, id_to_label = load_subjects(TARGET_LABELS)
    n_classes = len(id_to_label)
    print(f"Found {len(subjects)} subjects matching {TARGET_LABELS}.")

    # --- feature extraction ---
    print("Extracting embeddings using frozen LaBraM-Base …")
    all_embeddings, all_labels = [], []
    for subj in subjects:
        _, X_win, _ = load_window_pkl(subj["pkl"])
        h = embed_subject(model, X_win, input_chans, device)
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
            "run_id":        "LaBraM_linear_probe_benchmark",
            "created_at":    datetime.now().isoformat(),
            "phase":         "linear_probe_benchmark",
            "model":         "LaBraM-Base",
            "embed_dim":     200,
            "patch_size":    PATCH_SIZE,
            "target_labels": TARGET_LABELS,
            "n_subjects":    len(subjects),
            "n_channels_used": len(input_chans) - 1,
            "input_chans":   input_chans,
        },
        "global_metrics":           metrics,
        "global_metrics_std":       metrics_std,
        "global_metrics_formatted": formatted,
    }
    save_results(RESULTS_DIR, fold_rows, y_true, y_pred, n_classes, id_to_label, meta)

    print("=" * 78)
    print("LaBraM Linear Probing Benchmark DONE →", str(RESULTS_DIR))
    for k, v in formatted.items():
        print(f"  {k}: {v}")
    print("=" * 78)


if __name__ == "__main__":
    main()
