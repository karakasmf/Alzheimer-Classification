# Copilot instructions for Alzheimer-Classification 🔧💡

## Overview
This repo implements a REVE-based EEG classification pipeline for Alzheimer's Disease / FTD / Control classification using the **ds004504** dataset. Key stages:

```
build window cache → extract REVE embeddings → subject-level pooling → linear probe → evaluate
```

It also benchmarks three external EEG foundation models (LEAD, LaBraM, BIOT) under identical LOSO conditions.

---

## Quick Entry Points

| Task | Script |
|---|---|
| Preprocessing → window cache | `reve_pipeline/build_window_cache.py` |
| REVE Phase 1 (sklearn probe) | `reve_pipeline/runner/mode1_sklearn_loso.py` |
| REVE Phase 2 (end-to-end C2C) | `reve_pipeline/runner/end2end_phase2_c2cattn_loso.py` |
| LEAD benchmark | `benchmark/lead/run_lead_benchmark.py` |
| LaBraM benchmark | `benchmark/LaBraM/run_labram_benchmark.py` |
| BIOT benchmark | `benchmark/BIOT/run_biot_benchmark.py` |
| Unified results table | `analysis/analyze.py` → `results/summary_table.csv` |
| Grouped bar plot (all models) | `analysis/plot_grouped_bar.py` |

---

## Important File Formats & Conventions

- **Window cache** (`cache/windows/sub-XXX_windows.pkl`):  
  `payload = {"subject": str, "X": np.ndarray (n_win, n_ch, win_samp), "meta": dict}`  
  `meta` must include `ch_names` and `win_samp`. Loaded by `reve_pipeline/common/cache_io.py`.

- **Labels**: `participants.txt` — whitespace-delimited `<subject_id> <label_str>`.  
  Loaded with `reve_pipeline/common/labels.py`.

- **Results layout**: Each benchmark/run saves to its own `results/` folder containing:  
  `fold_results.csv`, `confusion_matrix.csv/.npy/.png`, `run_meta.json`  
  `run_meta.json` contains `meta`, `global_metrics`, `global_metrics_std`, `global_metrics_formatted`.

- **Centralized paths**: All project paths (WINDOW_DIR, RESULTS_ROOT, PARTICIPANTS_TXT) come from  
  `reve_pipeline/common/paths.py`. **Do not hardcode** these paths in scripts.

---

## Shared Benchmark Utilities

`benchmark/common/benchmark_utils.py` provides:
- `load_subjects(target_labels)` — loads subjects from window cache
- `run_loso_svm(X, y, subjects, id_to_label, seed)` — LOSO cross-validation with SVC
- `compute_metrics_bootstrapped(y_true, y_pred, n_classes, seed)` — metrics + bootstrap std
- `save_results(...)` — saves fold CSV, confusion matrix, PNG, run_meta.json

All three benchmark scripts import from this; do not duplicate this logic.

---

## Project-Specific Behaviors

- **Canonical channel order**: locked from the first valid subject in `build_window_cache.py`.  
  If `STRICT_CANONICAL=True`, subjects missing channels are skipped.

- **REVE embedding**: loaded via `transformers.AutoModel` with `trust_remote_code=True` in  
  `reve_pipeline/common/reve_embed.py`. Use `load_reve()`, `make_pos()`, `extract_window_embeddings()`.

- **Pooling**: `pool_subject(..., method='mean'|'trimmed_mean', trim_ratio=0.1)` in  
  `reve_pipeline/common/pooling.py`.

- **LOSO pattern**: one subject is held out as test set, remainder used for training.  
  Windows may be subsampled with `subsample(X, max_n, seed)` from `cache_io.py`.

- **Benchmark channel handling** (LaBraM/BIOT): pre-trained weights may be for different channel counts;  
  see channel mapping logic in each benchmark script.

---

## Runtime & Environment

- GPU: `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` in all runners.
- REVE models: `brain-bzh/reve-base`, `brain-bzh/reve-positions` from HuggingFace (`trust_remote_code=True`).
- Preprocessing: requires `mne`, reads EDF/SET/FIF/BDF.
- `timm==0.4.12` required for LaBraM.
- `linear-attention-transformer` required for BIOT.

---

## Conventions for New Experiments

1. Import shared paths from `reve_pipeline/common/paths.py` — never hardcode absolute paths.
2. Import LOSO/SVM/save logic from `benchmark/common/benchmark_utils.py` for new benchmarks.
3. Save results following the standard layout: `fold_results.csv`, `confusion_matrix.*`, `run_meta.json`.
4. Keep experimental knobs as module-level constants at the top of each runner script.