# Baseline comparison: RNN+flow vs. local-window predictors

**Date**: 2026-05-12
**Goal**: Compare the BiDirectionalMinGRU + NSF flow head against two distinct baseline tiers, on flux prediction as a function of offset `k`:
1. **Naive (closed-form) baselines** — `nn_mean`, `window_mean`. Zero parameters, no metadata, no learning. Establish "is the model doing anything?" floor. Sigma_k for NLL/coverage is fit offline on training residuals (one number per k, per baseline).
2. **Fair-budget learned baseline** — local-window MLP with metadata, learned per-point sigma, explicit k input. Architecture ablation: shares ~every fair advantage with the RNN except sequence integration, so RNN-vs-MLP isolates whether long-range integration matters.
3. **RNN + flow head** — the model under test.

The MLP is **not naive** — it has 50k+ params, metadata, and a learned σ. Don't conflate the two baseline tiers in the writeup.

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

**Converting point baselines to probabilistic**: for `nn_mean` and `window_mean`, fit a single `σ_k` per `k` on **training-set residuals** (disjoint from eval-10% and sanity-val) — one number per `(method, k)`. Use that fixed `σ_k` at eval time. This gives the naive baselines an honest probabilistic NLL without letting them cheat by adapting `σ` per-point. Implemented as the `fit_baselines` subcommand; writes `baseline_sigmas.json`.

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
├── mlp_gaussian_best.pt         # MLP context encoder + Gaussian head (early-stopped on sanity-val)
├── mlp_gaussian_last.pt         # last-epoch MLP checkpoint
├── mlp_gaussian_history.json    # per-epoch train + sanity-val loss; best_epoch
├── split.json                   # eval-10% gaia_ids + training gaia_ids (full set)
├── baseline_sigmas.json         # per-(method, k) Gaussian sigma for nn_mean / window_mean
├── summary.csv                  # per-(method, k): MAE, RMSE, NLL, coverage68, coverage95
├── mae_vs_k.png
├── nll_vs_k.png
└── rmse_vs_k.png
```

The MLP-NSF head is deprioritized — the MLP-Gaussian + naive-baseline-with-σ_k pair already answers the question. See "Decisions (continued)" below.

## Files to add

| File | Purpose |
|---|---|
| `scripts/baseline_comparison.py` | One-script entrypoint. Subcommands: `split`, `train_gaussian`, `fit_baselines`, `eval`, `plot`. (`train_nsf` deferred — see decisions below.) |
| `src/lcgen/models/mlp_baseline.py` | Shared context encoder + Gaussian head + plumbed NSF head. |
| `baseline_comparison.sh` | Shell wrapper: `split` + `train_gaussian`. |
| `baseline_comparison_eval.sh` | Shell wrapper: `fit_baselines` + `eval` + `plot`. |

## Decisions (locked in 2026-05-12)

- Accept the caveat that the RNN was trained on the eval-10%. Train the MLPs on 100% of stars too for symmetry.
- Include metadata in the MLP context vector.
- Add the NSF-headed MLP variant alongside the Gaussian one.

## Decisions (continued)

- **Sector handling**: per-sector eval (matches the flow's setup).
- **Training k-sampling**: one `k` per batch, log-uniform `[1, 720]` (matches `bounded_horizon_future_nll`). Eval uses a discrete `{1, …, 512}` grid for plotting only.
- **Sampling at training**: none. Both `mlp_nsf` and the existing RNN use NSF `log_prob` as the loss. Sampling is eval-only.
- **`n_flow_samples` at eval**: 128 (matches the notebook).

## Decisions (2026-05-12, post first eval)

- **Three-tier comparison locked in**: naive (closed-form) / fair-budget learned MLP / RNN+flow. The MLP is explicitly framed as the "architecture ablation" tier, not a naive baseline.
- **σ_k for naive baselines is fit on a training subsample**, not on eval residuals — keeps the comparison honest. ~500 training stars is enough for stable σ estimates at every k.
- **NSF-head MLP is deferred indefinitely.** The first eval pass showed `mlp_gaussian` and `window_mean` are essentially tied on MAE/RMSE — meaning the MLP encoder's gain over a window mean is small for point prediction. An NSF head on top would mostly demonstrate flow-vs-Gaussian on the SAME (weak) encoder, which is a different question from "does the RNN architecture matter". Revisit only if there's a specific reason to want it.
- **Eval n_flow_samples default is 0** (NLL via `log_prob` only). MAE-on-median + coverage for the flow require sampling and cost ~4× the runtime — opt in via the shell-script knob when needed.

## Remaining open question

- **Coverage runtime for the flow** — sampling 128 draws per (j, k) for the eval-10% would take ~2 hr on CPU. Either accept the cost or sample at a coarser j-grid. Decide if we end up wanting the coverage curves for the writeup.

## Out of scope

- Retraining the RNN.
- Comparing against GP baselines.
- Cross-method ensembling.
- Anything downstream of flux prediction (latents, age inference).
