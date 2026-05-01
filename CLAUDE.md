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
| `output/reconstructions/plot_reconstructions.ipynb` | Interactive notebook mirror of the above (reuses its helpers) |
| `kfold_age_inference.sh` | Shell wrapper for age inference |
| `slurm_bridges2.sh` | SLURM job script for PSC Bridges-2 |
| `sync_to_bridges2.sh` | Rsync project to Bridges-2 |
| `final_pretrain/all_ages.csv` | Age + photometry CSV (GaiaDR3_ID, age, BPRP0, BPRP0_err, MG_quick, mem_prob_val) |

## Conventions

- **Shell scripts: all parameters hardcoded.** Never pass arguments to `.sh` files — all config lives inside the script so parameters are always tracked and reproducible.
- **Two distinct age-prediction pipelines exist — don't conflate them:**
  1. `MetadataAgePredictor` — uses raw metadata (BPRP0, parallax, etc.) directly, just used for testing
  2. NLE flow in `kfold_age_inference.py` — uses autoencoder latent vectors
- **Always update this CLAUDE.md file** with the current status when changes are made, or when anything else in this file changes.
- **Always check for memory leaks when testing scripts.** This project uses a lot of data, and it is easy to crash my laptop. Make sure that large files aren't being loaded all at once or that scripts are not trying to store too much all in memory at once.
- **Always commit code/script/doc changes to git in logical chunks** before reporting a task done. Do NOT stage `.DS_Store`, `__pycache__/*.pyc`, `checkpoints/resume/*` (training state), or notebooks the user is actively editing — only commit source/script/doc changes from the current task. Group related changes into separate commits with descriptive messages, and never push without explicit user request.

## Model architecture (pretraining)

- **Model:** BiDirectionalMinGRU, 95,608 params (hidden_size=64, bi, parallel mode)
- **Parallel scan:** log-domain (`cumsum` + `logcumsumexp`), NOT the sequential loop
- **Activation:** `g(x) = relu(x) + 0.5` (non-negative, required for log-domain scan)
- **Flow head:** zuko NSF, 1D output conditioned on hidden state + time encoding + measurement error
- **Training loss:** `bounded_horizon_future_nll` — single randomly sampled k per batch (log-uniform from 1 to K)
- **Data:** ~60K samples across 2 H5 files, variable-length sequences (typically ~18,000 timesteps per sector)

## Age inference pipeline (`kfold_age_inference.py`)

Two encoder types for compressing autoencoder latents before the age flow:
1. **PCA** (`--encoder_type pca`): fixed PCA projection, no learnable params
2. **MLP** (`--encoder_type mlp`): learned encoder with auxiliary age L1 loss to prevent collapse

Age flow: NSF models `p(z | log10_age, BPRP0, log10(BPRP0_err), [log10(MG)])`.
Inference: likelihood on 1000-point grid, normalize to posterior, extract stats.

Star aggregation modes: `none`, `predict_mean`, `latent_mean`, `latent_median`, `latent_max`, `latent_mean_std`, `cross_sector`.

## Age inference model tracking

**Always update `final_model/final_parallel_e10/README.md`** when a new age inference model is trained or tested. The README contains a comparison table with MAE, Pearson r, and key hyperparameters for all runs. Regenerate metrics from the predictions CSVs in each `nf-*` subfolder.

Current best: 3-stage MLP, bottleneck_dim=4, finetune_encoder_lr_mult=0.001 (r=0.705, MAE=0.296 dex).

## Current status

- Pretraining complete with log-domain parallel scan. Active model checkpoints in `final_model/final_parallel_*`.
- Best age inference: 3-stage MLP encoder (r=0.705) — significantly better than PCA baseline (r~0.52).
- `--train_full` flag added to train a deployment model on all labeled stars after k-fold CV.
- `predict_ages.py` auto-detects `full_model.pt` for deployment; falls back to `kfold_models.pt` ensemble.
- Rolling checkpoint (`checkpoints/resume/`) saves latest epoch only; best model only saved to output dir at end of training.

## Bridges-2 (PSC)

- **Allocation:** `phy260003p`
- **Container:** `/ocean/containers/ngc/pytorch/delete/pytorch_24.11-py3.sif` (PyTorch 2.6, V100-compatible; newer containers dropped V100 sm_70 support)
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
