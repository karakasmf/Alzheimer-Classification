**Citation:** 


# Alzheimer Classification

> A Python-based EEG classification pipeline for Alzheimer's Disease (AD) vs. Frontotemporal Dementia (FTD) vs. Healthy Controls (CTL), benchmarking foundation models alongside a custom REVE-based approach.

> **Dataset:** [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) — *A dataset of EEG recordings from Alzheimer's disease, frontotemporal dementia, and healthy subjects.*

---

A rigorous comparative evaluation framework for EEG-based dementia classification. The project implements a **Leave-One-Subject-Out (LOSO)** cross-validation pipeline to benchmark multiple pre-trained EEG foundation models — **LEAD**, **LaBraM**, and **BIOT** — against a custom end-to-end architecture built on top of the **REVE** foundation model with a learnable Channel-to-Channel (C2C) attention module.

## Features

- **Foundation Model Benchmarking (Linear Probing)**:
  - Freezes each pre-trained backbone and extracts subject-level embeddings
  - Classifies with a standardized `StandardScaler + SVC` pipeline
  - Supports **LEAD** (NeurIPS 2023), **LaBraM** (ICLR 2024), and **BIOT** (NeurIPS 2023)
  
- **Custom REVE-based Pipeline**:
  - **Phase 1 (Baseline):** Frozen REVE encoder + sklearn Logistic Regression
  - **Phase 2 (Proposed):** Frozen REVE + trainable C2C Attention + linear head (end-to-end per-fold)

- **Rigorous Evaluation**:
  - LOSO cross-validation (65 subjects)
  - Bootstrap standard deviation (1 000 iterations) for all reported metrics
  - Confusion matrices, fold-level CSVs, and JSON run metadata
  - Permutation test for statistical significance between models

- **Reproducible Experiment Tracking**:
  - Each run auto-saves `run_meta.json` with all hyperparameters and metrics
  - Unified `analysis/analyze.py` generates a cross-model summary table

## Project Structure

```
Alzheimer-Classification/
│
├── reve_pipeline/
│   ├── build_window_cache.py        # EEG preprocessing → windowed .pkl cache
│   ├── common/
│   │   ├── paths.py                 # Centralized project paths (WINDOW_DIR, RESULTS_ROOT, …)
│   │   ├── cache_io.py              # Window cache loader utilities
│   │   ├── labels.py                # participants.txt reader
│   │   ├── metrics.py               # Accuracy, MCC, F1, sensitivity, specificity
│   │   ├── pooling.py               # Subject-level mean / trimmed-mean pooling
│   │   ├── reve_embed.py            # REVE model loading + window embedding extraction
│   │   └── c2c_attention.py         # Channel-to-Channel attention module
│   └── runner/
│       ├── mode1_sklearn_loso.py    # Phase 1: REVE + sklearn probe (baseline)
│       └── end2end_phase2_c2cattn_loso.py  # Phase 2: REVE + C2C + head (proposed)
│
├── benchmark/
│   ├── common/
│   │   └── benchmark_utils.py       # Shared LOSO-SVM, bootstrap, and save logic
│   ├── lead/
│   │   └── run_lead_benchmark.py    # LEAD linear probing benchmark
│   ├── LaBraM/
│   │   └── run_labram_benchmark.py  # LaBraM linear probing benchmark
│   └── BIOT/
│       └── run_biot_benchmark.py    # BIOT linear probing benchmark
│
├── analysis/
│   ├── analyze.py                   # Unified cross-model results summary table
│   ├── plot_grouped_bar.py          # Grouped bar chart (all models × all metrics)
│   ├── statistical_test.py          # Paired T-Test (baseline vs. proposed)
│   └── statistical_test_all.py      # Permutation test (all metrics)
│
├── cache/windows/                   # Auto-generated windowed EEG data (.pkl)
├── results/                         # REVE pipeline run outputs
└── requirements.txt
```

## Environment Setup

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Update dataset path** in `reve_pipeline/common/paths.py`:
   ```python
   PARTICIPANTS_TXT = Path(r"<path-to-ds004504>\derivatives\participants.txt")
   ```
   and in `reve_pipeline/build_window_cache.py`:
   ```python
   DERIVATIVES_ROOT = r"<path-to-ds004504>\derivatives"
   ```

## Usage

### Step 1 — Build the window cache (one-time preprocessing)
```bash
python reve_pipeline/build_window_cache.py
```
Outputs: `cache/windows/sub-XXX_windows.pkl` (one file per subject)

### Step 2 — Run a benchmark

**LEAD linear probing:**
```bash
python benchmark/lead/run_lead_benchmark.py
```

**LaBraM linear probing:**
```bash
python benchmark/LaBraM/run_labram_benchmark.py
```

**BIOT linear probing:**
```bash
python benchmark/BIOT/run_biot_benchmark.py
```

**REVE baseline (Phase 1 — sklearn probe):**
```bash
python reve_pipeline/runner/mode1_sklearn_loso.py
```

**REVE proposed (Phase 2 — end-to-end C2C):**
```bash
python reve_pipeline/runner/end2end_phase2_c2cattn_loso.py
```

### Step 3 — Analyse all results
```bash
python analysis/analyze.py             # → results/summary_table.csv
python analysis/plot_grouped_bar.py    # → results/grouped_metrics_barplot.png
python analysis/statistical_test.py   # → Paired T-Test
python analysis/statistical_test_all.py  # → Permutation test
```

## Results Summary (A vs C, LOSO, n=65)

| Model | Accuracy | Balanced Acc | MCC | Macro F1 |
|---|---|---|---|---|
| REVE Baseline (Phase 1) | — | — | — | — |
| REVE Proposed (Phase 2) | — | — | — | — |
| LEAD P-Base | — | — | — | — |
| LaBraM-Base | 0.754 ± 0.054 | — | 0.506 ± 0.109 | 0.752 ± 0.055 |
| BIOT (6-datasets) | 0.769 ± 0.053 | — | 0.532 ± 0.108 | 0.766 ± 0.054 |

*Run `analysis/analyze.py` to generate the full up-to-date table.*

## Foundation Models

| Model | Paper | Checkpoint |
|---|---|---|
| **LEAD** | NeurIPS 2023 | `benchmark/lead/checkpoints/` |
| **LaBraM** | ICLR 2024 | `benchmark/LaBraM/checkpoints/labram-base.pth` |
| **BIOT** | NeurIPS 2023 | `benchmark/BIOT/repo/pretrained-models/` |
| **REVE** | — | HuggingFace `brain-bzh/reve-base` |

## License

MIT


> Generative AI models were used to assist with code organization, refactoring, and review during the development process.