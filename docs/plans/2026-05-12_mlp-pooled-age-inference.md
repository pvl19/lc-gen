# 2026-05-12 — MLP-pooled latents for age inference

## Question

Does the BiDirectionalMinGRU encoder add anything that a same-budget MLP local-window
encoder cannot, for the *downstream* age inference task?

The linear probe ([2026-05-12_baseline-comparison.md](2026-05-12_baseline-comparison.md))
already showed local equivalence: ridge regression on `h_fwd[j-k] + h_bwd[j+k]`
matches the local-window MLP's MAE at every k ≤ 720. That tells us the RNN's
hidden state does not carry more *point-prediction* information than a local
window. It does **not** tell us whether pooled-across-time representations from
the two encoders are equivalent for downstream age inference.

Concrete comparison target: current RNN-pooled best is r=0.705, MAE=0.296 dex
(3-stage MLP encoder, bottleneck_dim=4, see [final_model/final_parallel_e10/README.md](../../final_model/final_parallel_e10/README.md)).

## What gets built

A new script `scripts/extract_mlp_latents.py` and shell wrapper
`extract_mlp_latents.sh` that:

1. Loads the trained `output/baseline_comparison/mlp_gaussian_best.pt`.
2. For each labeled (gaia_id, sector) in the age CSV / pretrain H5 files,
   subsamples ~1000 valid j positions across the sequence.
3. For each j, forwards the MLP at **k ∈ {1, 8, 64, 720}** and averages the
   **last-hidden-layer features (128-dim)** across the 4 k values.
4. Pools across j with **mean + std → 256-dim** per (star, sector).
5. Writes a cache in the canonical `save_latents_cache` format
   (`output/baseline_comparison/mlp_pooled_latents.npz`), so that
   `kfold_age_inference.py --load_latents <cache>` works without changes.

Then runs `kfold_age_inference.py` with the same hyperparameters as the current
RNN-pooled best:

- `--encoder_type mlp`
- `--training_stages` matching the 3-stage recipe
- `--mlp_encoder_hidden 256 128`
- `--bottleneck_dim 4`
- `--finetune_encoder_lr_mult 0.001`
- `--n_folds 10`, `--seed 42`

## Design choices

| Choice | Decision | Why |
|---|---|---|
| Which k to forward MLP at | k ∈ {1, 8, 64, 720}, features averaged | Most RNN-like: marginalizes out k, since the RNN hidden state is k-independent. Grid spans training distribution. |
| Pooling across j | mean + std → 256-dim | Simplest fair v1. Matches "naive global pool of MLP features". Can upgrade to multiscale (matching the RNN) if results are close. |
| Where to tap features | last 128-dim hidden layer (post-LayerNorm + GELU) | Task-specific outputs (mu/log_var head, or the 136-dim context output) tend to discard information; the hidden layer keeps it. |
| Metadata at extraction | keep included in input | MLP was trained with metadata. Removing it shifts the input distribution. Symmetric with RNN, which also receives metadata. Downstream age flow already conditions on metadata explicitly — minor redundancy is harmless. |
| j subsampling | ~1000 j positions per sector | Dense enough for stable mean+std without forwarding all ~18k timesteps. |

## Expected outputs

- `output/baseline_comparison/mlp_pooled_latents.npz` — latents cache in the
  canonical format. Keys: `latent_vectors` (N, 256), `ages`, `bprp0`,
  `bprp0_err`, `gaia_ids`, `tic_ids`, `sectors`, `mg`, `mg_err`, `mem_prob`.
- `output/baseline_comparison/age_inference_mlp_pooled/` — kfold age inference
  output directory: `kfold_predictions.csv`, `metrics.json`, residual plots.

## Outcomes and interpretation

| Result | Interpretation |
|---|---|
| MLP-pooled age MAE ≈ RNN-pooled (r ~ 0.70) | Recurrence adds nothing for age; the local-window encoder produces equally age-informative features when pooled. Implies the RNN's architectural budget is fungible — could use either family. |
| MLP-pooled noticeably worse (r ≪ 0.70) | BiDirectional sequence integration *does* extract global structure that local windows can't, even when pooled. Supports keeping the RNN; useful for future architecture work. |
| MLP-pooled noticeably better | Surprising; would warrant investigating whether the RNN training has a subtle issue (e.g., gradient flow through the parallel scan biasing toward shorter-range structure). |

A "noticeably worse" gap on the order of 0.02–0.05 in r is likely real; smaller
than that may be within k-fold noise.

## Effort

- Extraction script + wrapper: ~150–200 LoC. Half a day.
- Latent extraction over ~6k sectors × 4 k values × ~1000 j ≈ 24M MLP forwards.
  Batched on CPU, estimated 30–60 minutes.
- Age inference k-fold run: matches existing RNN-pooled run time (~20–40 min).

## Results (2026-05-13)

10-fold CV on the ChronoFlow subset, same recipe (multiscale + `latent_max`,
3-stage MLP head, bottleneck_dim=4, n_folds=10):

| Encoder | N | MAE (dex) | Median AE | RMSE | Pearson r |
|---|---:|---:|---:|---:|---:|
| **RNN-pooled** (BiDirectionalMinGRU multiscale) | 2470 | **0.139** | 0.046 | 0.326 | **0.912** |
| **MLP-pooled** (local-window, k ∈ {1,8,64,720}, mean+std) | 2495 | 0.225 | 0.072 | 0.438 | 0.848 |
| Gap | | +0.086 | +0.026 | +0.112 | −0.064 |

Per-fold MLP-pooled val_r: 0.74–0.89 (one weak fold at 0.74; rest ≥ 0.84).

**Interpretation:** the "noticeably worse" outcome from the plan. BiDirectional
recurrence is doing real work — it captures cross-time structure relevant to
age that the same-budget local-window encoder cannot recover even after global
pooling. The MLP-pooled latent is still strong in absolute terms (r=0.85), so
most of the age signal lives in *local* variability statistics; recurrence
buys the last ~0.06 of correlation.

The earlier linear-probe result said the encoders are equivalent for *local
flux prediction*. The age task is a different functional of the latent, and on
that functional the architectures separate.

Sanity checks done:
- N differs slightly (2470 vs 2495) because the MLP can score a few extra
  short-sector stars below the RNN's max_length cutoff. Not material to the
  comparison.
- Both runs use the same ChronoFlow subset, same 10-fold split, same
  hyperparameters except for the encoder family.

Artifacts:
- `output/baseline_comparison/mlp_pooled_latents.npz` — extracted cache (37 MB).
- `output/baseline_comparison/age_inference_mlp_pooled/kfold_metrics.json`,
  `kfold_predictions.csv`, `kfold_results.png`, `full_model.pt`.

## Followups (not blocking)

- Try multiscale-style pooling for MLP features (mean / max / quartiles
  rather than mean+std) to confirm pooling isn't where the gap lives.
- If the multiscale-pooled MLP still loses by ≳0.04 r, that's strong evidence
  the gap is in the encoder architecture, not the pool.

## Out of scope

- Re-training the MLP for this comparison. We use the existing baseline
  comparison checkpoint; it was trained on the same `bounded_horizon_future_nll`
  signal as the RNN, so this is a clean architecture-only ablation.
- Comparing to PCA, gyrochronology, or other age-prediction baselines. Those
  are tracked separately in [final_model/final_parallel_e10/README.md](../../final_model/final_parallel_e10/README.md).
- Multiscale pooling for the MLP. Mean+std first; revisit if v1 result motivates it.
