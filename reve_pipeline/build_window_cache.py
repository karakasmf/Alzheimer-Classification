#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reve_pipeline/build_window_cache.py
(Moved from project root: build_window_cache_reve_paper.py)

REVE-paper-aligned window cache builder (ds004504 derivatives).

Paper-aligned preprocessing:
- Keep recordings >= 10 seconds
- Resample to 200 Hz
- Band-pass filter 0.5–99.5 Hz
- Convert amplitude to microvolts
- Recording/session-level Z-score normalization (per-channel)
- Clip to ±15 standard deviations
- Enforce canonical channel order across all subjects

Output: cache/windows/sub-XXX_windows.pkl
"""

import os
import sys
import glob
import pickle
import logging
from typing import List, Optional
from pathlib import Path

import numpy as np
import mne

# Resolve paths from paths.py so OUT_DIR is always correct regardless of CWD
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from reve_pipeline.common.paths import WINDOW_DIR

# =========================
# CONFIG (Edit these)
# =========================
DERIVATIVES_ROOT = r"D:\ACADEMICS\datasets\alz-ftd-ctl\ds004504\derivatives"
OUT_DIR          = str(WINDOW_DIR)   # uses paths.py → cache/windows/

# Windowing
WIN_SEC    = 2.0
STEP_SEC   = 1.0
MAX_WINDOWS = None   # e.g. 2000 or None

# Paper-aligned preprocessing
TARGET_SFREQ = 200.0
BANDPASS     = (0.5, 99.5)
NOTCH        = None
REREF        = None

MIN_TOTAL_SEC = 10.0
ZCLIP_STD     = 15.0
OVERWRITE     = True
VERBOSE       = True

STRICT_CANONICAL   = True
CHANNEL_RENAME_MAP = {}

SUPPORTED_EXTS = (".set", ".fif", ".edf", ".bdf")


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("build_window_cache")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def find_eeg_file(subject_dir: str) -> Optional[str]:
    for ext in SUPPORTED_EXTS:
        files = glob.glob(os.path.join(subject_dir, "**", f"*{ext}"), recursive=True)
        if files:
            return files[0]
    return None


def load_raw(path: str) -> mne.io.BaseRaw:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".set":
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    if ext == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    if ext == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if ext == ".bdf":
        return mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    raise RuntimeError(f"Unsupported EEG format: {path}")


def apply_preprocessing(raw: mne.io.BaseRaw, logger: logging.Logger) -> mne.io.BaseRaw:
    raw.pick("eeg")
    if CHANNEL_RENAME_MAP:
        present = {ch: CHANNEL_RENAME_MAP[ch] for ch in raw.ch_names if ch in CHANNEL_RENAME_MAP}
        if present:
            raw.rename_channels(present)
            logger.debug(f"Renamed channels: {present}")
    if BANDPASS is not None:
        raw.filter(l_freq=BANDPASS[0], h_freq=BANDPASS[1], verbose="ERROR")
    if NOTCH is not None:
        raw.notch_filter(NOTCH, verbose="ERROR")
    if REREF is not None:
        if REREF.lower() == "average":
            raw.set_eeg_reference("average", verbose="ERROR")
        else:
            logger.warning(f"Unknown reref='{REREF}'. Skipping.")
    if abs(float(raw.info["sfreq"]) - TARGET_SFREQ) > 1e-6:
        raw.resample(TARGET_SFREQ, npad="auto", verbose="ERROR")
    return raw


def reorder_to_canonical(
    data: np.ndarray, ch_names: List[str], canonical: List[str],
    subject: str, logger: logging.Logger,
) -> Optional[np.ndarray]:
    idx = {c.strip(): i for i, c in enumerate(ch_names)}
    canonical_clean = [c.strip() for c in canonical]
    missing = [c for c in canonical_clean if c not in idx]
    if missing:
        msg = f"{subject} missing channels: {missing}"
        if STRICT_CANONICAL:
            logger.warning("[SKIP] " + msg)
            return None
        logger.warning("[FILL-ZERO] " + msg)
    out = np.zeros((len(canonical_clean), data.shape[1]), dtype=np.float32)
    for i, c in enumerate(canonical_clean):
        if c in idx:
            out[i] = data[idx[c]]
    return out


def recording_zscore_and_clip(x: np.ndarray, clip_std: float) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    z  = (x - mu) / sd
    if clip_std is not None:
        z = np.clip(z, -float(clip_std), float(clip_std))
    return z.astype(np.float32, copy=False)


def sliding_window(x: np.ndarray, win_samp: int, step_samp: int) -> np.ndarray:
    n_ch, n_samp = x.shape
    if n_samp < win_samp:
        return np.empty((0, n_ch, win_samp), dtype=np.float32)
    n_win = 1 + (n_samp - win_samp) // step_samp
    out   = np.empty((n_win, n_ch, win_samp), dtype=np.float32)
    for i in range(n_win):
        s = i * step_samp
        out[i] = x[:, s: s + win_samp]
    return out


def main():
    logger = setup_logger(VERBOSE)
    os.makedirs(OUT_DIR, exist_ok=True)

    subjects = sorted([d for d in os.listdir(DERIVATIVES_ROOT) if d.startswith("sub-")])
    logger.info(f"Found {len(subjects)} subjects under derivatives.")

    canonical_ch_names: Optional[List[str]] = None
    ok, skipped = 0, 0

    for sub in subjects:
        subj_dir  = os.path.join(DERIVATIVES_ROOT, sub)
        out_path  = os.path.join(OUT_DIR, f"{sub}_windows.pkl")

        if os.path.exists(out_path) and not OVERWRITE:
            logger.info(f"[SKIP] cache exists: {sub}"); skipped += 1; continue

        eeg_file = find_eeg_file(subj_dir)
        if eeg_file is None:
            logger.warning(f"[SKIP] no EEG file: {sub}"); skipped += 1; continue

        try:
            raw = load_raw(eeg_file)
            raw = apply_preprocessing(raw, logger)
        except Exception as e:
            logger.warning(f"[SKIP] load/preprocess failed: {sub} | {e}")
            skipped += 1; continue

        sfreq        = float(raw.info["sfreq"])
        data_v       = raw.get_data().astype(np.float32)
        duration_sec = data_v.shape[1] / sfreq

        if duration_sec < MIN_TOTAL_SEC:
            logger.warning(f"[SKIP] too short: {sub} ({duration_sec:.1f}s)")
            skipped += 1; continue

        if canonical_ch_names is None:
            canonical_ch_names = list(raw.ch_names)
            logger.info(f"[CANONICAL] locked from {sub}: {len(canonical_ch_names)} ch")
            logger.info(f"[CANONICAL] {canonical_ch_names}")

        data_v = reorder_to_canonical(data_v, raw.ch_names, canonical_ch_names, sub, logger)
        if data_v is None:
            skipped += 1; continue

        data_uv   = data_v * 1e6
        data_norm = recording_zscore_and_clip(data_uv, ZCLIP_STD)
        win_samp  = int(WIN_SEC * sfreq)
        step_samp = int(STEP_SEC * sfreq)
        X         = sliding_window(data_norm, win_samp, step_samp)

        if X.shape[0] == 0:
            logger.warning(f"[SKIP] insufficient samples: {sub}")
            skipped += 1; continue

        if MAX_WINDOWS is not None and X.shape[0] > int(MAX_WINDOWS):
            X = X[: int(MAX_WINDOWS)]

        payload = {
            "subject": sub,
            "X": X.astype(np.float32, copy=False),
            "meta": {
                "sfreq": sfreq, "target_sfreq": TARGET_SFREQ,
                "bandpass": BANDPASS, "win_sec": WIN_SEC, "step_sec": STEP_SEC,
                "win_samp": win_samp, "step_samp": step_samp,
                "n_channels": int(X.shape[1]),
                "ch_names": list(canonical_ch_names),
                "duration_sec": float(duration_sec),
                "source_file": eeg_file,
                "strict_canonical": bool(STRICT_CANONICAL),
                "zscore_scope": "recording_per_channel",
                "clip_std": float(ZCLIP_STD),
                "units": "zscore_clipped_(V→uV)",
            },
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[OK] {sub} | windows={X.shape[0]} | ch={X.shape[1]}")
        ok += 1

    logger.info(f"DONE | ok={ok} | skipped={skipped}")
    if canonical_ch_names is None:
        logger.error("No subject processed. Check DERIVATIVES_ROOT.")


if __name__ == "__main__":
    main()
