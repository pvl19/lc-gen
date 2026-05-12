# Baseline comparison: RNN+flow vs. local-window predictors

**Date**: 2026-05-12
**Goal**: Show that the BiDirectionalMinGRU + NSF flow head predicts flux better than naive local-window baselines (interpolation, mean) and a fair-budget learned baseline (MLP on a local window). Quantify both point accuracy (MAE, RMSE) and probabilistic quality (NLL, coverage) as a function of prediction offset `k`.

## What gets (re)trained

| Component | Action |
|---|---|
| `BiDirectionalMinGRU` + flow head | **Not retrained.** Load `final_model/parallel_fixed/e110/best_model.pt` as-is. |
| MLP baseline — Gaussian head | **New.** Trained from scratch on **all** stars (100%), conditioned on `k`. |
| MLP baseline — NSF head | **New.** Same MLP context encoder, NSF flow output instead of Gaussian. Also trained on 100% of stars. |
| Non-learned baselines (NN-mean, linear interp, window mean) | **No training.** Closed-form evaluation. |

**Why train the MLPs on 100% of stars rather than 90%**: the existing RNN was pretrained on the full dataset (no held-out split). Training the MLPs on the same 100% puts both learned methods at the same epistemic position w.r.t. the eval set. The held-out 10% then functions as a *tractable, deterministic eval subset* rather than a true held-out test set — neither model has truly unseen data. This is documented as a caveat in the writeup. The split key still exists so a future fully-honest comparison (RNN retrained on 90%) is a drop-in.

**Caveat to record in the writeup**: both the RNN and the MLP baselines were trained on the full set of stars. The reported metrics on the "held-out" 10% are therefore not generalization metrics in the strict sense — they are in-distribution metrics on stars all models have seen during training. A fully-honest test would require retraining all three learned methods on a 90% split.

## Data split

- **Source**: `final_pretrain/timeseries_pretrain.h5` + `final_pretrain/timeseries_exop_hosts.h5` (same as Section B/B2/C in `plot_reconstructions.ipynb`).
- **Split key**: `GaiaDR3_ID`. Stars (not sectors) are assigned to train/eval so a star never crosses splits.
- **Split mechanism**: deterministic — `hash(str(gaia_id)) % 10 == 0` puts a star in the eval 10%. Single hash, no per-run randomness, so the split is reproducible across scripts and notebooks.
- **MLP training set**: all stars *except* the 1% sanity-val subset (see below). The eval-10% serves as a tractable, deterministic eval subset; it is **included** in the MLP training set to mirror the RNN's exposure.
- **Sanity-val split**: a deterministic 1% of stars (`hash(str(gaia_id)) % 100 == 7`) held out of MLP training *only*. Purpose: monitor for overfitting and pick the epoch count. **Not** used in the final comparison metrics. Disjoint from the eval-10% by construction.
- The split keys are written to `split.json` so a future RNN-retrained-on-90% comparison can reuse them without code changes.

## MLP baseline architecture

**Shared context encoder** for both Gaussian and NSF variants:

- **Input** (concatenated, fixed-length vector):
  - `flux[j-C-k : j-k]` — C points just before the gap on the forward side (1D, length C)
  - `flux[j+k : j+k+C]` — C points just after the gap on the backward side (length C)
  - `flux_err` for those same 2C points (length 2C)
  - Relative times `(t - t_j)` for those 2C points (length 2C)
  - Valid-mask channel for the 2C window (length 2C; handles edge padding)
  - `flux_err[j]` — the target point's measurement error (length 1)
  - `log2(k)` — offset encoding, lets a single MLP cover all `k` (length 1)
  - Metadata vector (length 13, same `METADATA_FEATURES` the RNN's `meta_encoder` consumes)
- **Context encoder**: 2 hidden layers of 128 units, GELU, LayerNorm. Outputs a `D=64` context vector matched to the RNN's hidden size for a fair NSF comparison.
- **Window size**: `C = 32`.
- **Edge handling**: pad with zeros, set the valid-mask channel to 0 for padded positions.

**Two output heads, trained as separate models**:

1. **Gaussian head** (`mlp_gaussian`): linear projection of the context vector to `(μ, log σ²)`. Trained with Gaussian NLL.
2. **NSF head** (`mlp_nsf`): the same zuko NSF used in the existing flow head, conditioned on `concat(context_vector, time_encoding(t_j), flux_err[j])` — exactly the conditioning input the RNN's flow head receives. Trained with NSF NLL.

The `mlp_nsf` variant is the apples-to-apples test of the encoder: same flow head, different context vector source (RNN hidden state vs. windowed MLP). The `mlp_gaussian` variant is the simpler "is the flow head doing meaningful work?" test.

**Size budget**: each variant aims for ~50–100k total params, matched to the flow head + final RNN layer for fairness. Final exact size set once context encoder + head are written.

## Training procedure

Two separate training runs — one per head — using the same data loader and `(j, k)` sampling.

- **Data loader**: reuse `TimeSeriesDataset` to load sequences; a thin wrapper samples `(j, k)` pairs within each sequence, builds the fixed-length context vector, returns `(input, target_flux)`.
- **k-sampling**: **one `k` per batch** (not per sample), drawn log-uniform from `[1, K_max=720]` — exactly matches the RNN's `bounded_horizon_future_nll` training distribution. All samples in a batch share the same `k`, simplifying the fixed-size context construction. The discrete eval grid `{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}` is used only for plotting, not for training.
- **Target**: `flux[j]` at the chosen point (un-masked — the model never sees `flux[j]`, only neighbours).
- **Training stars**: all stars except the 1% sanity-val subset. The eval-10% *is* in the training set (matches the RNN's exposure).
- **Loss**: Gaussian NLL on `(μ, log σ²)` for `mlp_gaussian`; NSF NLL for `mlp_nsf`.
- **Optimizer**: Adam, lr=1e-3, cosine decay. Tentative ~5 epochs; refine after watching sanity-val loss.
- **Sanity-val monitoring**: compute mean loss on the 1% sanity-val set after each epoch. Use to pick best-epoch checkpoint and detect overfit. Does not enter the final comparison.
- **Batch size**: 256.
- **Compute**: CPU is fine for `mlp_gaussian`; GPU recommended for `mlp_nsf`. Single workstation; no need for Bridges-2.
- **Outputs**:
  - `output/baseline_comparison/mlp_gaussian.pt`
  - `output/baseline_comparison/mlp_nsf.pt`
  - Each pickle stores model + config + train args.

## Evaluation protocol

For each method (`rnn_flow`, `mlp_nsf`, `mlp_gaussian`, `nn_mean`, `linear_interp`, `window_mean`) and each star in the eval 10%:

1. Load the sequence.
2. For each `k` in `{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}`:
   - Skip if `length <= 2k`.
   - Predict `flux[j]` for every valid `j` in `[k, length-k)`.
   - Record: per-point `(μ, σ, target)` (for probabilistic methods) or `(prediction, target)` (for point methods).
3. Aggregate: per-star MAE, RMSE, NLL (Gaussian for baselines, sample-based for flow), 68%/95% interval coverage.

**Converting point baselines to probabilistic**: for `nn_mean`, `linear_interp`, `window_mean`, fit a single `σ_k` per `k` on the training split's residuals (one number per `k`). Use that fixed `σ_k` at eval time. This gives them an honest NLL without letting them cheat by adapting `σ` per-point.

**Sampling vs. log_prob (flow methods, eval only)**:

| Quantity | Flow methods (`rnn_flow`, `mlp_nsf`) | Cost |
|---|---|---|
| NLL | `log_prob` (analytic) | cheap |
| Median / p16 / p84 for MAE & coverage | sample `n_flow_samples=128`, take quantiles | dominates eval time |
| Posterior band plots (notebook) | same sampling path | same |

The RNN's `bounded_horizon_future_nll` training loss is `log_prob`-based — no sampling at training time. Sampling appears only at eval/inference. Same applies to `mlp_nsf`.

**Eval k-grid**: discrete `{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}` for plotting. The flow models extrapolate fine to any k within the training range `[1, 720]`.

## Metrics & plots

- **Aggregate table**: rows = method, cols = MAE / RMSE / NLL / 68% coverage / 95% coverage, averaged across held-out stars at a representative `k` (e.g. `k=1` and `k=64`).
- **MAE vs k**: one curve per method, log-x, with IQR band across stars. Expected: flow flattest, baselines fall off sharper at large `k`.
- **NLL vs k**: same axes. Expected: flow wins more here than on MAE because its `σ` adapts per-point.
- **Coverage vs k**: 68% and 95% empirical coverage. Calibrated methods sit near the nominal line.

## Outputs

```
output/baseline_comparison/
├── mlp_gaussian.pt              # MLP context encoder + Gaussian head
├── mlp_nsf.pt                   # MLP context encoder + NSF head
├── split.json                   # eval-10% gaia_ids + training gaia_ids (full set)
├── predictions.parquet          # per-(star, j, k, method): mu, sigma, target
├── summary.csv                  # per-(method, k): mean MAE, RMSE, NLL, coverage + IQR
├── mae_vs_k.png
├── nll_vs_k.png
└── coverage_vs_k.png
```

`predictions.parquet` lets us re-aggregate later without re-running inference.

## Files to add

| File | Purpose |
|---|---|
| `scripts/baseline_comparison.py` | One-script entrypoint. Subcommands: `split`, `train_gaussian`, `train_nsf`, `eval`, `plot`. Each step writes its outputs and can be re-run independently. |
| `src/lcgen/models/mlp_baseline.py` | Shared context encoder + Gaussian head + NSF head. |
| `bin/baseline_comparison.sh` (or `baseline_comparison.sh` at repo root, matching existing convention) | Shell wrapper, all params hardcoded. |

## Decisions (locked in 2026-05-12)

- Accept the caveat that the RNN was trained on the eval-10%. Train the MLPs on 100% of stars too for symmetry.
- Include metadata in the MLP context vector.
- Add the NSF-headed MLP variant alongside the Gaussian one.

## Decisions (continued)

- **Sector handling**: per-sector eval (matches the flow's setup).
- **Training k-sampling**: one `k` per batch, log-uniform `[1, 720]` (matches `bounded_horizon_future_nll`). Eval uses a discrete `{1, …, 512}` grid for plotting only.
- **Sampling at training**: none. Both `mlp_nsf` and the existing RNN use NSF `log_prob` as the loss. Sampling is eval-only.
- **`n_flow_samples` at eval**: 128 (matches the notebook).

## Remaining open question

- **Coverage runtime** — full resolution is ~200 stars × 10 k-values × ~18k points × 128 samples for the two flow methods. If this is too slow, subsample timesteps per sector at eval (e.g. every 10th point). Decide once we measure.

## Out of scope

- Retraining the RNN.
- Comparing against GP baselines.
- Cross-method ensembling.
- Anything downstream of flux prediction (latents, age inference).
