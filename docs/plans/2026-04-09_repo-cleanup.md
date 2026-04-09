# Repo Cleanup Plan

**Date:** 2026-04-09
**Status:** Pending

## Context

The repo root has 30 shell scripts, loose .py files, stale docs, and miscellaneous artifacts. The goal is to organize without changing any core functionality. All Python scripts in `scripts/` and `src/lcgen/` stay put.

## What stays at root

These 4 actively-used shell scripts remain at root:
- `kfold_age_inference.sh`
- `plot_umap.sh`
- `slurm_bridges2.sh`
- `sync_to_bridges2.sh`

Plus: `CLAUDE.md`, `README.md`, `requirements.txt`, `.gitignore`

## Moves

### 1. Shell scripts → `bin/` (with subdirs)

**`bin/train/`** — training launchers:
- `train_model.sh`, `train_model_bridges2.sh`, `train_age_predictor.sh`, `train_full_nle.sh`, `run_all_pooling_modes.sh`

**`bin/inference/`** — inference scripts (not actively used):
- `kfold_gyro_baseline.sh`, `kfold_metadata_age.sh`, `kfold_prot_inference.sh`, `kfold_ps_acf_prediction.sh`, `kfold_transformer_age.sh`, `predict_ages.sh`

**`bin/plot/`** — visualization:
- `plot_recon.sh`, `replot_residuals.sh`, `replot_umap.sh`, `vis_time_enc.sh`, `compare_age_models.sh`, `compare_from_csv.sh`, `compare_pooling_modes.sh`

**`bin/hpc/`** — remote cluster (not actively used):
- `copy_to_bridges2.sh`, `copy_to_bebop.sh`, `copy_to_expanse.sh`, `copy_checkpoint_to_expanse.sh`, `slurm_expanse.sh`, `test_bridges2.sh`, `setup_resume_checkpoint.sh`

**`bin/data/`** — data preparation:
- `convert_timeseries_to_h5.sh`

### 2. Stale docs → `docs/`

- `CONV_CHANNELS_SUMMARY.md` → `docs/`
- `EXPANSE_FOLDER_STRUCTURE.md` → `docs/`
- `RESUME_QUICK_REFERENCE.md` → `docs/`
- `RESUME_TRAINING.md` → `docs/`

### 3. Loose root files → appropriate homes

- `test_ddp_smoke.py` → `tests/`
- `benchmark_conv_speed.py` → `tests/`
- `count_model_params.py` → `tests/`
- `adhoc.ipynb` → `notebooks/`
- `loss_comparison.png` → `paper_figs/`
- Delete `src/lcgen/models/simple_min_gru.py.bak`

### 4. Data notebooks → `notebooks/data_prep/`

These are currently gitignored under `data/*` — need to use plain `mv` then `git add`:
- `data/download_lightcurves.ipynb` → `notebooks/data_prep/`
- `data/download_lightcurves_p2.ipynb` → `notebooks/data_prep/`
- `data/get_DR2_from_TICs.ipynb` → `notebooks/data_prep/`
- `data/parallax_zeropoint_corrections.ipynb` → `notebooks/data_prep/`
- `data/raw_data_load.ipynb` → `notebooks/data_prep/`
- `data/reddening_calc_with_dist_err.ipynb` → `notebooks/data_prep/`
- `data/search_TICS_from_names.ipynb` → `notebooks/data_prep/`
- `data/search_lightkurve.ipynb` → `notebooks/data_prep/`

### 5. Add `__init__.py` files

- `src/lcgen/__init__.py` (empty)
- `src/lcgen/models/__init__.py` (empty)
- `src/lcgen/utils/__init__.py` (empty)

### 6. Delete stale dirs

- `legacy/` — empty, delete

### 7. Keep as-is

- `ilay_sims/` — will be used going forward
- All of `scripts/`, `src/lcgen/`, `notebooks/`, `data/`, `output/`, `final_model/`, `final_pretrain/`, `checkpoints/`, `exop_hosts/`, `paper_figs/`

## Cross-references to update

After moves, these files have internal path references that need fixing:

1. **`sync_to_bridges2.sh`** (stays at root) — line 39 rsyncs `slurm_bridges2.sh` (stays at root) → no change needed
2. **`run_all_pooling_modes.sh`** (moving to `bin/train/`) — calls `./kfold_age_inference.sh` → update to use repo-root-relative path
3. **`copy_to_bridges2.sh`** (moving to `bin/hpc/`) — references `slurm_bridges2.sh` → update path
4. **`test_bridges2.sh`** (moving to `bin/hpc/`) — references `slurm_bridges2.sh` → update path
5. **`CLAUDE.md`** — update "Important files" table with new paths for moved scripts

## Verification

1. `git status` — confirm all moves tracked, no unintended changes
2. Run `./kfold_age_inference.sh` — confirm it still works (stayed at root)
3. Run `./sync_to_bridges2.sh` — confirm rsync paths still resolve
4. `python -c "import src.lcgen"` — confirm __init__.py doesn't break imports
5. Check no shell script fails due to broken cross-references
