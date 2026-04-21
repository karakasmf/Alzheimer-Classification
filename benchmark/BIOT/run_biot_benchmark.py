#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIOT Linear Probing Benchmark
================================
Uses BIOT (NeurIPS 2023) as a frozen feature extractor.
LOSO cross-validation with SVM classifier (Linear Probing).

Shared LOSO/SVM/save logic lives in benchmark/common/benchmark_utils.py.
Only BIOT-specific model loading and embedding extraction is here.

Channel handling:
  The best checkpoint (EEG-six-datasets-18-channels.ckpt) was pre-trained
  with 18 bipolar channels.  We load all weights EXCEPT channel_tokens,
  which is re-initialised for our 19 unipolar channels (first 18 rows
  copied from checkpoint, 19th row is random init).
"""

import sys
import random
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BENCHMARK_DIR = Path(__file__).parent.parent          # benchmark/
_PROJECT_ROOT  = _BENCHMARK_DIR.parent                 # alz-ftd-ctl-reve/
_BIOT_REPO     = Path(__file__).parent / "repo"

for p in [str(_PROJECT_ROOT), str(_BENCHMARK_DIR), str(_BIOT_REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    from model.biot import BIOTEncoder
except Exception as e:
    print(f"Dependency error loading BIOT: {e}")
    sys.exit(1)

from common.benchmark_utils import (
    load_subjects, run_loso_svm, compute_metrics_bootstrapped, save_results
)
from reve_pipeline.common.cache_io import load_window_pkl

# ===========================================================================
# CONFIG
# ===========================================================================
TARGET_LABELS   = ["A", "C"]
SEED            = 42
RESULTS_DIR     = Path(__file__).parent / "results"

EMB_SIZE        = 256
N_HEADS         = 8
DEPTH           = 4
N_FFT           = 200
HOP_LENGTH      = 100
N_OUR_CHANNELS  = 19
CKPT_NAME       = "EEG-six-datasets-18-channels.ckpt"


# ===========================================================================
# Model loading
# ===========================================================================

def load_frozen_biot(ckpt_path: Path, device: str) -> BIOTEncoder:
    model = BIOTEncoder(
        emb_size=EMB_SIZE, heads=N_HEADS, depth=DEPTH,
        n_channels=N_OUR_CHANNELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    ).to(device)

    print(f"Loading BIOT checkpoint from {ckpt_path} …")
    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    own_state   = model.state_dict()
    loaded, skipped = [], []
    for k, v in state_dict.items():
        if k not in own_state:
            skipped.append(k); continue
        own_shape = own_state[k].shape
        if v.shape == own_shape:
            own_state[k] = v; loaded.append(k)
        elif k == "channel_tokens.weight":
            n_ckpt = v.shape[0]
            own_state[k][:n_ckpt, :] = v
            loaded.append(f"{k} [partial {n_ckpt}→{own_shape[0]}]")
        elif k == "index":
            skipped.append(f"{k} (re-init for {N_OUR_CHANNELS} channels)")
        else:
            skipped.append(f"{k} shape mismatch {v.shape} vs {own_shape}")

    model.load_state_dict(own_state)
    print(f"  Loaded {len(loaded)} tensors. Skipped/re-init: {len(skipped)}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("BIOT loaded and frozen.")
    return model


# ===========================================================================
# Feature extraction
# ===========================================================================

def embed_subject(model: BIOTEncoder, X_win: np.ndarray,
                  device: str) -> np.ndarray:
    """(N, C, T) → (256,) subject-level embedding."""
    model.eval()
    batch_size = 64
    embs = []
    with torch.no_grad():
        for s in range(0, X_win.shape[0], batch_size):
            batch = torch.from_numpy(
                X_win[s:s + batch_size].astype(np.float32)
            ).to(device)
            emb = model(batch)   # (B, 256)
            embs.append(emb.cpu().numpy())
    all_emb = np.concatenate(embs, axis=0)   # (N, 256)
    return np.mean(all_emb, axis=0)          # (256,)


# ===========================================================================
# Main
# ===========================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BIOT Linear Probing Benchmark | Device: {device}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ckpt_path = Path(__file__).parent / "repo" / "pretrained-models" / CKPT_NAME
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    model    = load_frozen_biot(ckpt_path, device)
    subjects, id_to_label = load_subjects(TARGET_LABELS)
    n_classes = len(id_to_label)
    print(f"Found {len(subjects)} subjects matching {TARGET_LABELS}.")

    # --- feature extraction ---
    print(f"Extracting embeddings using frozen BIOT ({CKPT_NAME}) …")
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
            "run_id":        "BIOT_linear_probe_benchmark",
            "created_at":    datetime.now().isoformat(),
            "phase":         "linear_probe_benchmark",
            "model":         "BIOT",
            "checkpoint":    CKPT_NAME,
            "emb_size":      EMB_SIZE,
            "n_fft":         N_FFT,
            "hop_length":    HOP_LENGTH,
            "target_labels": TARGET_LABELS,
            "n_subjects":    len(subjects),
            "n_channels":    N_OUR_CHANNELS,
        },
        "global_metrics":           metrics,
        "global_metrics_std":       metrics_std,
        "global_metrics_formatted": formatted,
    }
    save_results(RESULTS_DIR, fold_rows, y_true, y_pred, n_classes, id_to_label, meta)

    print("=" * 78)
    print("BIOT Linear Probing Benchmark DONE →", str(RESULTS_DIR))
    for k, v in formatted.items():
        print(f"  {k}: {v}")
    print("=" * 78)


if __name__ == "__main__":
    main()
