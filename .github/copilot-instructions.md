# Copilot instructions for alz-ftd-ctl-reve 🔧💡

Overview
- This repo implements a REVE-based EEG classification pipeline focused on LOSO (leave-one-subject-out) experiments. Key stages: build window cache -> extract per-window REVE embeddings -> subject-level pooling -> linear probes (sklearn or torch) -> evaluate and save run metadata.

Quick entry points
- Build cached windows (preprocessing + windowing): `build_window_cache_reve_paper.py` (see top for config constants: DERIVATIVES_ROOT, OUT_DIR, WIN_SEC, TARGET_SFREQ, etc.). Output: `cache/windows/sub-<id>_windows.pkl`.
- Run experiments (examples): `reve_pipeline/runner/mode1_sklearn_loso.py` (sklearn probe), `mode2_torchprobe_loso.py`, `phase2_c2cattn_loso.py` (C2C attention ablation/phase2), `end2end_phase2_c2cattn_loso.py`.
- Summarize results: `reve_pipeline/analyze.py` (creates `results/summary_table.csv`).

Important file formats & conventions
- Window cache pickle format (created by `build_window_cache_reve_paper.py`): payload = {"subject": str, "X": np.ndarray (n_win, n_ch, win_samp), "meta": dict}. `meta` must include `ch_names` and `win_samp`.
- Cache loader: `reve_pipeline/common/cache_io.py` expects files matching `sub-*_windows.pkl` under `cache/windows`.
- Labels: `participants.txt` is whitespace-delimited lines: `<subject_id> <label_str>`. Load with `reve_pipeline/common/labels.py`.
- Results layout: runs are saved under `results/<TAG>/run_*`. Each run contains `fold_results.csv`, `confusion_matrix.csv/.npy`, training logs `training_log*.csv`, and `run_meta.json` with `meta` (see `RunMeta` dataclasses in runner scripts) and `global_metrics`.

Project-specific behaviors to follow
- Canonical channel order: The cache builder locks a canonical channel order from the first valid subject (see `build_window_cache_reve_paper.py`). If `STRICT_CANONICAL=True`, subjects missing any canonical channels are skipped. Modify `CHANNEL_RENAME_MAP` to handle vendor naming differences.
- Embedding extraction: REVE + positions are loaded using `transformers.AutoModel` with `trust_remote_code=True` in `reve_pipeline/common/reve_embed.py`. The positional bank expects a canonical list of channel names. Use `make_pos` and `extract_window_embeddings` utilities.
- Pooling choices: `pool_subject(..., method='mean'|'trimmed_mean', trim_ratio=0.1)` (see `reve_pipeline/common/pooling.py`). Experiments rely on subject-level pooling after per-window embeddings.
- LOSO fold pattern: Runners iterate folds by leaving one subject out as test set. Training subject windows may be subsampled with `subsample(..., max_n, seed)` (see `cache_io.py`).

Runtime & environment notes
- GPU-enabled code selects `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` in runners. Ensure CUDA and PyTorch are set up if you expect GPU runs.
- Preprocessing requires `mne` and reading EDF/SET/FIF/BDF; `build_window_cache_reve_paper.py` uses MNE and will error on unsupported formats.
- Transformers models used: `brain-bzh/reve-base` and `brain-bzh/reve-positions`. `trust_remote_code=True` is necessary; be aware of remote model code trust implications.

Debugging & developer workflows
- Quick local debug: reduce `MAX_WINDOWS`, `MAX_TRAIN_WINDOWS_PER_SUBJ`, and `BATCH_SIZE` to run faster on a single machine.
- Per-fold training logs are written to `training_log_foldXXX.csv` under the run directory. Use these to inspect epoch-level behavior.
- If `load_window_pkl` raises `meta['ch_names'] missing`, rebuild cache with the builder script; this is a common issue when cache was created incorrectly.
- To reproduce a run, re-run the same runner script with the same top-level constants (`TARGET_LABELS`, `SEED`, model names). `run_meta.json` records most hyperparameters.

Conventions for contributors and new experiments
- Follow the runner pattern: create a `RunMeta` dataclass, save `fold_results.csv`, `confusion_matrix.*`, and `run_meta.json` containing `meta` and `global_metrics`.
- Keep top-level experimental knobs as module-level constants (as in existing runner scripts) so runs are reproducible and easily tweaked.
- Use `reve_pipeline/common/*` utilities for loading, embedding extraction, pooling, metrics.

Where to look for examples
- End-to-end example (sklearn probe & LOSO): `reve_pipeline/runner/mode1_sklearn_loso.py` ✅
- C2C module: `reve_pipeline/common/c2c_attention.py` (shows module interface and expected input shapes) ✅
- Cache builder & canonical-channel logic: `build_window_cache_reve_paper.py` ✅
- Summary utility: `reve_pipeline/analyze.py` ✅

Security and caveats
- Transformers models are loaded with `trust_remote_code=True`. Exercise caution in environments with strict security policies.
- Absolute paths are used in config constants (Windows-oriented). Replace with env vars or relative paths for portability.

If anything above is unclear or you'd like more examples (e.g., common refactor patterns, how to run an end-to-end debug session), tell me which area to expand and I'll iterate. ✅