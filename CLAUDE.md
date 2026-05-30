# CLAUDE.md

## Project overview

Light curve autoencoder for stellar age inference from TESS 2-min cadence data.
BiDirectional MinGRU encodes variable-length light curves into latent representations;
a separate normalizing flow (NLE) predicts stellar ages from the latent space via
k-fold cross-validation.

## Key directories

- `src/lcgen/` — core library (model, dataset, loss, utils)
- `scripts/` — inference, plotting, data prep Python scripts
- `bin/` — shell script launchers organized by category (`train/`, `inference/`, `plot/`, `hpc/`, `data/`)
- `final_pretrain/` — training data (H5 files + age CSV)
- `final_model/` — trained model checkpoints + cached latents
- `checkpoints/resume/` — rolling checkpoint for Bridges-2 resume (latest epoch, NOT best)
- `notebooks/` — analysis notebooks + `data_prep/` subfolder for data munging notebooks
- `docs/` — archived documentation + `plans/` for implementation plans (dated, tracked in git)
- `tests/` — benchmarks and smoke tests

## Important files

| File | Purpose |
|------|---------|
| `src/lcgen/models/simple_min_gru.py` | BiDirectionalMinGRU with log-domain parallel scan |
| `src/lcgen/train_simple_rnn.py` | Main pretraining script (DDP, torchrun) |
| `src/lcgen/utils/loss.py` | `bounded_horizon_future_nll` — training loss |
| `scripts/kfold_age_inference.py` | Age prediction: PCA or MLP encoder → NSF flow |
| `scripts/plot_umap_latent.py` | Latent extraction, `load_ages()`, UMAP plotting |
| `scripts/plot_reconstructions.py` | Flow-based reconstructions at a given offset (random / Gaia ID / TIC ID), output to `output/reconstructions/` |
| `plot_reconstructions.sh` | Shell wrapper for reconstructions |
| `scripts/extract_split_latents_s97_98.py` | Re-extract latents for sectors 97/98 with each ~60-day light curve split in half; writes sidecar npz with extra `subsector`/`orig_h5_idx` fields, does NOT overwrite the main latents |
| `extract_split_latents_s97_98.sh` | Shell wrapper for the above (default model: sendit/e100) |
| `scripts/merge_split_latents_s97_98.py` | Drop sector 97/98 rows from a main latents npz and replace with the per-half split rows; writes a new merged npz (does NOT overwrite). Main + split must come from the same model checkpoint. |
| `merge_split_latents_s97_98.sh` | Shell wrapper: merges the three (pretrain/hosts/thickdisk) banks into `*_merged.npz` under the same `LATENT_BASE` |
| `scripts/split_sector_stats_s97_98.py` | Build a new `data/sector_stats_s97s98_split.csv` mirroring the latent split: sector-97/98 rows split into two halves with recomputed `flux_skew`/`flux_kurt`; adds a `subsector` column (-1 unchanged, 0/1 halves) that joins to the merged latents npz. Other stats copied unchanged. |
| `scripts/update_sector_stats_umap.py` | After a fresh UMAP run on the merged latents bank, populate `umap_x`/`umap_y` in the split sector_stats CSV via a (gaia_id, sector, subsector) join. Verifies row-by-row alignment between the UMAP npz and the concatenated merged latents before joining. |
| `output/reconstructions/plot_reconstructions.ipynb` | Interactive notebook mirror of the above (reuses its helpers) |
| `src/lcgen/models/mlp_baseline.py` | MLP local-window baseline (Gaussian + plumbed NSF heads) |
| `scripts/baseline_comparison.py` | RNN-vs-baseline comparison: `split` / `train_gaussian` subcommands |
| `baseline_comparison.sh` | Shell wrapper: split + train Gaussian MLP baseline |
| `baseline_comparison_eval.sh` | Shell wrapper: eval all methods on the eval-10% split |
| `kfold_age_inference.sh` | Shell wrapper for age inference |
| `kfold_sector_robust.sh` | Shell wrapper: sector-confound-robust age inference (3 mitigations) |
| `slurm_bridges2.sh` | SLURM job script for PSC Bridges-2 |
| `sync_to_bridges2.sh` | Rsync project to Bridges-2 |
| `final_pretrain/all_ages.csv` | Age + photometry CSV (GaiaDR3_ID, age, BPRP0, BPRP0_err, MG_quick, mem_prob_val) |

## Conventions

- **Shell scripts: all parameters hardcoded.** Never pass arguments to `.sh` files — all config lives inside the script so parameters are always tracked and reproducible.
- **`--trim_edges` must match between training and inference.** Default is 10 — strips the first/last 10 raw samples from every light curve before the encoder sees them, to avoid TESS sector-edge artifacts (scattered light, thermal settling). Set in `slurm_bridges2.sh` (training), `plot_umap.sh` (latent extraction), `predict_ages.sh`, `plot_reconstructions.sh`. The model itself is unchanged — only the data slice fed to it. Samples with `length < 2*trim_edges + 32` are dropped at the dataset level. Old checkpoints trained with `trim_edges=0` are NOT compatible with `trim_edges=10` inference (and vice versa) — the edge data is OOD for the model that didn't see it.
- **Two distinct age-prediction pipelines exist — don't conflate them:**
  1. `MetadataAgePredictor` — uses raw metadata (BPRP0, parallax, etc.) directly, just used for testing
  2. NLE flow in `kfold_age_inference.py` — uses autoencoder latent vectors
- **Always update this CLAUDE.md file** with the current status when changes are made, or when anything else in this file changes.
- **Always check for memory leaks when testing scripts.** This project uses a lot of data, and it is easy to crash my laptop. Make sure that large files aren't being loaded all at once or that scripts are not trying to store too much all in memory at once.
- **Always commit code/script/doc changes to git in logical chunks** before reporting a task done. Do NOT stage `.DS_Store`, `__pycache__/*.pyc`, `checkpoints/resume/*` (training state), or notebooks the user is actively editing — only commit source/script/doc changes from the current task. Group related changes into separate commits with descriptive messages, and never push without explicit user request.

## Model architecture (pretraining)

- **Model:** BiDirectionalMinGRU, 95,608 params (hidden_size=64, bi, parallel mode)
- **Parallel scan:** log-domain (`cumsum` + `logcumsumexp`), NOT the sequential loop
- **Recurrence gating:** masked/padded positions (`mask == 0`) are hard-gated out of the minGRU scan — the update gate is forced to 0 so `h_t = h_{t-1}`. The step contributes nothing to the recurrence and gets zero gradient; gating a block is equivalent to deleting it. Applies in both `step_parallel` and `step`; the backward scan uses the flipped mask. Verified by `tests/test_recurrence_gating.py`.
- **Activation:** `g(x) = relu(x) + 0.5` (non-negative, required for log-domain scan)
- **Flow head:** zuko NSF, 1D output conditioned on hidden state + time encoding + measurement error
- **Metadata masking (DOROTHY-style):** when `--meta_use_mask` is set, the metadata encoder takes an explicit binary mask channel (input dim 13→26) and `DynamicMetadataMasking` (`src/lcgen/utils/metadata_masking.py`) applies train-only masking — per-star whole-encoder block-drop `Bernoulli(--meta_block_mask_prob)` plus DOROTHY-style per-field masking (`p_keep ~ U(--meta_keep_min, --meta_keep_max)`, guaranteed-keeper) over all 13 fields. A robustness regularizer; it does NOT remove the sector confound (which is Route B — light-curve-derived; see `project_sector_confound`). Verified by `tests/test_metadata_masking.py`.
- **Training loss:** `bounded_horizon_future_nll` — single randomly sampled k per batch (log-uniform from 1 to K). K=2880 (≈4 days) in the final run, matched to `--max_size`.
- **Data:** ~60K samples across 2 H5 files, variable-length sequences (typically ~18,000 timesteps per sector)

## Age inference pipeline (`kfold_age_inference.py`)

Two encoder types for compressing autoencoder latents before the age flow:
1. **PCA** (`--encoder_type pca`): fixed PCA projection, no learnable params
2. **MLP** (`--encoder_type mlp`): learned encoder with auxiliary age L1 loss to prevent collapse

Age flow direction (`--prediction_mode`):
- `nle` (default): NSF models `p(z | log10_age, BPRP0, log10(BPRP0_err), [log10(MG)])`; the age posterior is recovered by Bayes-inverting a likelihood grid over the age context.
- `npe`: NSF conditions on the bottleneck **+** colours and outputs age directly — `p(log10_age | z, BPRP0, log10(BPRP0_err), [log10(MG)])` — so the prediction is read off the flow's own 1D density (the grid only discretizes that output pdf for percentiles, no Bayes inversion). Supported for both learned encoders (`AgePredictorNPE`, mlp/linear) **and** the fixed PCA projection (`AgePredictorPCANPE` — PCA features fed straight in as flow context, no learned encoder). `predictions.csv` format is identical across modes.
  - **NPE spline cap → target standardization.** Because NPE makes age the *modelled* variable, it passes through the zuko NSF's hard ±5 spline support (`MonotonicRQSTransform` bound=5). Raw Gyr ages (0–15) overflow it and saturate at a ~5 Gyr ceiling. Both `AgePredictorPCANPE` and `AgePredictorNPE` (mlp/linear) therefore z-score the age target by one **global** `(loc, scale)` (computed over all labeled stars in `run_kfold_cv`, shared across every fold + the deployment `full_model`, stored as model buffers, inverted at predict time → predictions stay in input units, no manual re-standardization on reload). Driven by `--npe_standardize_target`/`--no-npe_standardize_target` (default on for NPE; no-op for NLE). On hosts (sendit/e50 PCA4, gyr, EIV) lifting the cap took r 0.359→0.501, MAE 1.60→1.41 Gyr. NLE is immune — there age is *context*, not the modelled variable.

### Host age-inference knobs (`kfold_age_inference_hosts.sh` + `scripts/kfold_nle_age_inference_hosts.py`)

The canonical configurable host launcher; all toggles hardcoded inside the `.sh`. Beyond mode/encoder/age-space:
- **`BALANCE_AGE` (+ `N_BALANCE_AGE_BINS`, `BALANCE_AGE_TEMP`).** `WeightedRandomSampler` flattens the training age marginal so NPE stops collapsing weakly-conditioned predictions onto the data mode (~4 Gyr). Weight ∝ (1/bin_count)^T (T=1 fully flattens, <1 softens the sparse old tail, 0=off). Sector-free age-only balance (`balance_age` in `train_single_fold`), distinct from the sector×age `balance_sector_age`. De-biases / matches the true marginal at the cost of point r (bias↔variance).
- **Learned MLP/linear encoder over the full 1536-d latent** (`ENCODER_TYPE=mlp`): the PCA cache is skipped; `MLP_ENCODER_HIDDEN`/`AUX_LOSS_WEIGHT`/`DROPOUT`/`INPUT_DROPOUT`/`VARIANCE_REG_WEIGHT` apply. Aux age head + variance reg prevent bottleneck collapse (no spiky-cluster-age memorization risk for continuous host ages — that was the pretrain/cluster concern). `INPUT_DROPOUT` masks raw latent dims pre-encoder (stronger regularizer than hidden dropout for the wide input; `_indrop<p>` output-dir suffix).
- **EIV now works for pca AND mlp/linear** (joint-only). The flow consumes the learnable latent age; the MLP aux head keeps targeting the fixed `y_obs` via `forward(..., aux_log_age=)` so they can't co-adapt and defeat the collapse guard. Shell forces `TRAINING_STAGES=joint` when EIV is on.
- **Empirical ceiling:** every host variant (pca4/16, mlp4, ±EIV, ±balance, ±input-dropout) lands at r≈0.42–0.45 — an information ceiling set by how much age the latents encode for field hosts; architecture only trades bias/variance/calibration around it.

Inference: likelihood on 1000-point grid, normalize to posterior, extract stats.

Star aggregation modes: `none`, `predict_mean`, `latent_mean`, `latent_median`, `latent_max`, `latent_mean_std`, `cross_sector`.

### Generalization / sector-confound validation modes

Three mutually-exclusive CV schemes probe out-of-distribution generalization (all read the GLOBAL r/MAE over held-out predictions, not per-fold r):

- **`--loocv_age`** — leave-one-cluster-out (ChronoFlow proxy): unique isochrone ages ≈ clusters; `loocv_age_folds()` bundles the young end, keeps big clusters as individual folds, bundles sparse ages. Tests cluster generalization / anti-memorization.
- **`--sector_level_split`** — per-sector sector-disjoint CV (use with `--star_aggregation none`/`predict_mean`): holds out whole sectors per fold; drops val rows whose star also appears in a training sector (star-disjoint).
- **`--loso`** — leave-one-sector-out, **per-star** (`--star_aggregation latent_max`). `loso_sector_folds()` partitions unique sectors into `--n_folds` disjoint groups; fold f holds out group f. VALIDATION = stars assigned to f (each validated once, in a group its sectors touch); TRAINING = only stars whose sectors are **all** outside group f — every star touching a held-out sector is removed from training entirely (whole star, not just its held-out-sector rows). Each star's sector-set is threaded through `latent_max` aggregation (rebuilt in `main` from `gaia_ids`→`sectors`, aligned to the aggregated rows). **Folds are balanced** (deterministic, not random): sectors are partitioned LPT-style by star-load (even training sizes) and stars are assigned to their least-loaded touched group (even validation sizes) — on the ~11.5K labeled pretrain stars / 99 sectors this gives val spread ≈1 and train ≈71% (vs random's 615–2133 / 6304–9891). PCA is fit per-fold on training stars only (no projection leakage). Wrapper: `kfold_loso.sh` (picks PCA dim from the dim-sweep's best LOCO r, fallback 8; all labeled pretrain stars, cache ages). Verified by `tests/test_sector_mitigations.py`.

## Age inference model tracking

**Always update `final_model/final_parallel_e10/README.md`** when a new age inference model is trained or tested. The README contains a comparison table with MAE, Pearson r, and key hyperparameters for all runs. Regenerate metrics from the predictions CSVs in each `nf-*` subfolder.

Current best: 3-stage MLP encoder, bottleneck_dim=4, multiscale + latent_max aggregation, ChronoFlow subset (r=0.912, MAE=0.139 dex on N=2470 stars). See `final_model/parallel_fixed/e110/age-inference-cfonly-multiscale-latent_max/kfold_metrics.json`.

## Current status

- Pretraining complete with log-domain parallel scan. Active model checkpoints in `final_model/final_parallel_*`.
- Best age inference: r=0.912, MAE=0.139 dex (RNN-pooled multiscale + 3-stage MLP head).
- `--train_full` flag added to train a deployment model on all labeled stars after k-fold CV.
- `predict_ages.py` auto-detects `full_model.pt` for deployment; falls back to `kfold_models.pt` ensemble.
- Rolling checkpoint (`checkpoints/resume/`) saves latest epoch only; best model only saved to output dir at end of training.
- Baseline-comparison pipeline: Gaussian-head MLP trainer + chunk-aware H5 loader + `eval` / `plot` / `fit_baselines` / `linear_probe` subcommands. Compares `mlp_gaussian`, `rnn_flow`, `nn_mean`, `window_mean` on the eval-10% set; reports NLL (mean / clipped / per-seq median) + MAE + RMSE + coverage. See `docs/plans/2026-05-12_baseline-comparison.md`.
- MLP-pooled age inference comparison: same-budget local-window MLP encoder, pooled mean+std across time, run through the same 3-stage age flow → r=0.848, MAE=0.225 dex on N=2495. RNN-pooled beats it by Δr=0.064, ΔMAE=0.086 dex — recurrence carries age-relevant cross-time structure that local-window pooling can't recover. See `docs/plans/2026-05-12_mlp-pooled-age-inference.md`.
- **Multiscale pool is now time-aware (Route A).** `compute_multiscale_features` requires a `t_valid` argument: `glob_mean/std` are Voronoi-time-weighted, segments split `[t_min, t_max]` into equal-time bins, `diff_*` operate on rates Δh/Δt (unweighted mean/std). Order statistics (`glob_max/min`, `first_h`, `last_h`) unchanged. Motivation: the pre-existing pool ignored the irregular time axis the encoder is conditioned on, which created discrete UMAP islands for 60-day sectors (97/98) and large-gap sectors (notably s77 where `diff_mean` z=+4.78). All callers (`plot_umap_latent.py`, `predict_ages.py`, `kfold_age_inference.py` per-sector + cross-sector) pass `t_valid` now. Cross-sector concatenation uses `_concat_sectors_with_time` to stitch per-sector times onto a synthetic continuous axis (no inter-sector gap injected). **Existing cached latents (`final_model/parallel_fixed/e110_*/latents_*.npz`) and any age inference models trained on them are obsolete and need regeneration / retraining**; `diff_*` blocks especially shift scale because they are now `hidden/day` rather than `hidden/step`.
- **Final pretraining run — robustness changes (`docs/plans/2026-05-18_final-pretraining-run.md`).** Three changes, all implemented and unit-tested: (1) hard recurrence gating of masked/padded positions; (2) `max_size = K = 2880` (4 days) in `slurm_bridges2.sh`; (3) DOROTHY-style metadata masking via `--meta_use_mask --meta_block_mask_prob 0.15 --meta_keep_min 0.3 --meta_keep_max 1.0`. `RESUME_FROM` cleared — fresh run (Changes 1/3 break checkpoint compatibility: the metadata encoder input dim goes 13→26). After this run all cached latents and age models are obsolete and must be regenerated. The D7 ablation confirmed the sector confound is Route B (light-curve-derived) — this run improves robustness but is **not** expected to drop the sector probe.
- **Sector-robust age inference (`docs/plans/2026-05-22_sector-robust-age-inference.md`).** Three age-inference-stage mitigations for applicability to non-cluster stars, all in `kfold_age_inference.py` (require per-sample sectors → `--star_aggregation none`/`predict_mean`), driven by `kfold_sector_robust.sh`, tested in `tests/test_sector_mitigations.py`: (1) `--sector_level_split` — sector-disjoint CV (hold out whole sectors per fold; star-overlap dropped by default; NaN-safe metrics) as a field-star generalization diagnostic; (2) `--balance_sector_age` — WeightedRandomSampler flattening the (sector × age-bin) joint; (3) `--adv_sector_weight` — GRL/DANN sector adversary on the MLP bottleneck (λ ramped, encoder-training only, discarded at inference; requires `--encoder_type mlp`). Tune the adversary on the in-pipeline sector-disjoint generalization, NOT the input-latent partial-r probe (it reshapes the bottleneck, which isn't exported).
- **Final sendit model (`final_model/sendit/e50/best_model.pt`).** e50 with metadata masking + the log10-metadata-flux fix (no adversarial training). Latents for all stars extracted to `final_model/sendit/e50/metaAll/latents_{pretrain,hosts,thickdisk}.npz` via `plot_umap.sh` (locked-in pooling: uniform/equal_count/dt). LOCO finding on the prior e110 latents: `mlp_random` r=0.913 → `mlp_d4` LOCO r=0.560 (memorization gap), `pca_d8` LOCO r=0.710 (PCA generalizes out-of-cluster better than the supervised MLP), `gyro` LOCO r=0.498 → going forward with **PCA** for the age head.
- **PCA dim sweep + LOSO validation (staged on sendit/e50).** `kfold_pca_dimsweep.sh` (pca{4,8,16} LOCO + one random-10-fold ceiling) picks the PCA dim by best out-of-cluster (LOCO) r; `kfold_loso.sh` then runs leave-one-sector-out (`--loso`, all labeled pretrain stars, PCA at the sweep-chosen dim) → out-of-sector generalization on the deployed pipeline. Both run sequentially after latent extraction (one heavy job at a time).

## Bridges-2 (PSC)

- **Allocation:** `phy260003p`
- **Container:** `/ocean/containers/ngc/pytorch/pytorch_24.11-py3.sif` (PyTorch 2.6, V100-compatible; newer containers dropped V100 sm_70 support). PSC relocated this out of a former `delete/` subdir — re-check the path if the job fails with "could not open image".
- **GPU partition:** `GPU-shared` with `v100-32:4` (4× V100-32GB)
- **A100 access:** `GPU-dev` partition only, limited to 1× A100. No multi-GPU A100 partition visible.
- **torch.compile:** skipped on V100 (requires sm_75+); works on A100 in GPU-dev
- **Sync:** `sync_to_bridges2.sh` rsyncs source code + SLURM script only (data files are already on Bridges-2 and excluded from sync — they are too large)
- **SLURM copies `src/` to local SSD scratch** — after code changes, re-sync and resubmit

## Commands

```bash
# Sync to Bridges-2
./sync_to_bridges2.sh

# Submit training job
sbatch slurm_bridges2.sh

# Run age inference locally
./kfold_age_inference.sh
```
