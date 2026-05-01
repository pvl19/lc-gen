---
date: 2026-04-30
status: IMPLEMENTED
---

# K-fold MLP age regression for exoplanet hosts

## 1. Motivation

`kfold_age_inference.py` predicts ages by training a normalizing-spline-flow
posterior `p(z | log10_age, BPRP0, log10(BPRP0_err))` and inverting it on a
1000-point log-age grid. The flow gives well-calibrated posteriors but is
overkill when the goal is simply a point estimate, and the multi-stage training
schedule (encoder pretrain → flow → joint fine-tune) is sensitive to LR /
optimizer state.

This script provides a baseline: a plain MLP regressor that maps the same input
features directly to `log10(age_Myr)`. It serves as both a sanity check on what
the flow buys us and a faster-to-iterate alternative for downstream uses that
don't need full posteriors.

## 2. Pipeline

1. **Load** cached host latents from `exop_host_ages/latents.npz` (or any other
   cache). The loader accepts both schemas:
   - kfold/UMAP format (`latent_vectors, ages, bprp0, ...`)
   - `predict_ages.py` format (`latents, gaia_ids, bprp0, [bprp0_err]`)
   Missing fields (ages, BPRP0, BPRP0_err, MG_quick) are joined from
   `archive_ages_default.csv` + `host_all_metadata.csv` by GaiaDR3_ID.
2. **Filter** rows with NaN in any required field (age, BPRP0, BPRP0_err,
   plus MG if `--use_mg`).
3. **Aggregate** per-sector latents to one row per unique GaiaDR3_ID using one
   of `latent_mean / latent_median / latent_max / latent_mean_std` (matches
   the modes in `kfold_age_inference.py`).
4. **Build features**: latents, optionally concatenated with `BPRP0`,
   `log10(BPRP0_err)`, and `log10(MG_quick)`.
5. **Stratified K-fold** by log-age quantile bins (default 10 bins, 10 folds).
   For each fold:
   - Standardize features using train-fold mean/std (test fold uses the same
     stats — no leakage).
   - Train a fresh MLP from scratch with cosine LR + early stopping.
   - Predict on the held-out fold.
6. **Aggregate** predictions across folds and write CSV + plots + metrics.
7. **(optional)** With `--train_full`, train one deployment model on ALL stars
   (with a 10% random hold-out for early stopping) and save `full_model.pt`.

## 3. MLP architecture

Plain feedforward MLP (`AgeRegressorMLP` in
[scripts/kfold_mlp_age_inference.py](../../scripts/kfold_mlp_age_inference.py)):

```
x (in_dim) ─► [Linear → GELU → Dropout]  (hidden_dims[0])
            ─► [Linear → GELU → Dropout]  (hidden_dims[1])
            ─► …
            ─► Linear → ŷ  (1)
```

Defaults (overridable from the shell script):

| Knob              | Default          | Notes                                       |
|-------------------|------------------|---------------------------------------------|
| `mlp_hidden`      | `[256, 128, 64]` | Three hidden layers, 64-dim bottleneck      |
| Activation        | GELU             | Same family as the flow encoder for parity  |
| `dropout`         | 0.1              | Applied after each hidden activation        |
| `loss_fn`         | MSE              | `huber` (SmoothL1) available for robustness |
| Optimizer         | AdamW            | `weight_decay=1e-4`                         |
| LR schedule       | CosineAnnealingLR| `T_max = n_epochs`, no warmup               |
| `lr`              | `1e-3`           | Peak LR                                     |
| `n_epochs`        | 300              | Hard cap — early stop usually fires first   |
| `patience`        | 40               | Epochs without val-loss improvement         |
| `batch_size`      | 64               | Same as the flow pipeline                   |

**Input dim** depends on aggregation + metadata:

| Aggregation         | Latent contribution | + metadata (`use_metadata`) | + MG          |
|---------------------|---------------------|-----------------------------|----------------|
| `latent_max` (etc.) | `D` (e.g. 1536)     | `D + 2`                     | `D + 3`        |
| `latent_mean_std`   | `2D`                | `2D + 2`                    | `2D + 3`       |

Feature standardization (zero-mean / unit-variance) is fit on each train fold
and applied to the matching val fold. The `(mean, std)` per fold is saved in
`kfold_models.pt`.

**Target**: `log10(age_Myr)`. Predictions are returned both in log-space and
exponentiated to Myr for the predictions CSV.

## 4. Stratification

Folds are built with `StratifiedKFold` over **log-age quantile bins** (10 bins
by default). This keeps each fold's age distribution similar to the global one
— important since hosts are not uniformly distributed in age. Stratification
operates on per-star rows after aggregation, so all sectors of a star stay in
the same fold (no leakage).

## 5. Outputs

Written to `--output_dir`:

| File                       | Contents                                                                 |
|----------------------------|--------------------------------------------------------------------------|
| `predictions.csv`          | `GaiaDR3_ID, true/pred log10(age), true/pred age_Myr, BPRP0, BPRP0_err, fold` |
| `scatter_pred_vs_true.png` | Predicted vs. true log10(age) with the y=x line, MAE/RMSE/r in the title |
| `training_curves.png`      | Per-fold train + val loss curves                                         |
| `metrics.json`             | Overall metrics, per-fold metrics, full CLI config                       |
| `kfold_models.pt`          | List of fold state-dicts + per-fold `(mean, std)` normalizers            |
| `full_model.pt`            | Single deployment model trained on all stars (when `--train_full`)       |

## 6. Files added

- [scripts/kfold_mlp_age_inference.py](../../scripts/kfold_mlp_age_inference.py)
- [kfold_mlp_age_inference.sh](../../kfold_mlp_age_inference.sh) — hardcoded
  shell wrapper (project convention: never pass args to `.sh` files).

## 7. Comparison vs. flow pipeline

| Property            | NSF flow (`kfold_age_inference.py`) | MLP (this script)                   |
|---------------------|-------------------------------------|-------------------------------------|
| Output              | Posterior `p(log10_age \| z, …)`    | Point estimate `ŷ = log10(age)`     |
| Loss                | Bounded-horizon NLL + aux L1        | MSE (or Huber)                       |
| Encoder             | PCA / MLP encoder + flow head       | Direct: features → MLP → scalar     |
| Stages              | 1 / 2 / 3-stage                     | Single stage                        |
| Uncertainty         | Full posterior, calibration metrics | None (could ensemble or add MC dropout) |
| Compute             | Heavier (per-grid-point likelihoods)| Light (single forward pass)         |
| Use case            | Calibrated science result           | Fast iteration / sanity baseline    |

## 8. Open questions / extensions

- **Uncertainty quantification.** A k-fold ensemble already exists in
  `kfold_models.pt`; predictions can be averaged across the K trained MLPs at
  inference time to get a rough sigma. MC-dropout or deep ensembles would be
  cleaner if uncertainty becomes important.
- **Hyperparameter search.** Defaults are reasonable but unsearched. Worth
  sweeping `mlp_hidden`, `dropout`, and `loss_fn=huber` once the baseline is
  characterized.
- **Pretrain-on-non-hosts variant.** Mirroring `kfold_combined_age_inference`,
  the MLP could be pretrained on the labeled non-host pretrain set and
  fine-tuned per host fold. Not implemented; flagged for later if the simple
  baseline plateaus too far below the flow.
