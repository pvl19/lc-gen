# Masked-Region Collapse Analysis

**Date:** 2026-04-27
**Symptom:** With `--mode parallel`, the trained autoencoder + flow head predicts ~0 flux in masked regions even after training with `mask_portion=0.5` and the new mask-aware loss (sources re-routed to nearest unmasked positions, masked targets contributing to loss).

## TL;DR

There is a **real, severe numerical bug** in the parallel scan's `log_g` activation that makes the parallel mode compute hidden states that diverge from the intended sequential dynamics by a factor of ~150 wherever `W_h(x) < 0`. The code path

```python
torch.where(x >= 0, (F.relu(x)+0.5).log(), 5 - F.softplus(-x))
```

contains a spurious additive constant `5` in the negative branch. The correct expression for `log(sigmoid(x))` (which is what `g(x) = sigmoid(x)` gives in the sequential code) is `-F.softplus(-x)`. The constant `5` makes `exp(log_g(x)) ≈ 148 × g(x)` for all `x < 0`.

This bug **alone** could explain the parallel-vs-sequential divergence the user remembers, and is the single most likely cause of the masked-region collapse. It also predicts how the model adapts during training: it learns to keep `W_h(x) ≥ 0` to avoid the buggy branch, which collapses the cell's nonlinearity to a near-affine `g(x) = x + 0.5` and starves the model of expressive dynamics — exactly the recipe for "constant prediction inside large mask blocks."

A secondary, architectural contributor (independent of the bug) is that with the new source-remap loss, **all targets inside a single mask block share the same forward and backward source hidden states**. Inside a mask block of width >> k, only `t_enc` varies between predictions, so the flow's predictions inside the block can vary only as much as `t_enc` informs them. If time conditioning is weak, the prediction collapses to the dataset's prior mean (≈ 0 for normalized light curves).

## Symptoms

- Train + val loss converged to ~1.11 (after our patches: source remapping in loss, masked targets included in loss, padding exclusion via `times != 0`).
- Visual reconstruction plots: in unmasked regions the flow median tracks the data; in masked regions the flow median is essentially flat at flux ≈ 0 with very narrow p16–p84 uncertainty bands.
- Behavior persists at multiple offsets `k` and across multiple example light curves.
- The user has anecdotally noted the **sequential** mode did not behave this way (no specific saved example).

## What's actually happening — root-cause analysis

### 1. The `log_g` bug (most likely root cause)

[src/lcgen/models/simple_min_gru.py:40-42](src/lcgen/models/simple_min_gru.py#L40-L42):

```python
@staticmethod
def log_g(x):
    return torch.where(x >= 0, (F.relu(x)+0.5).log(), 5 - F.softplus(-x))
```

**What this is supposed to compute.** The parallel scan represents the recurrence
`h_t = (1 - σ(k_t)) · h_{t-1} + σ(k_t) · g(W_h x_t)` in log space. To do that, every quantity that ends up inside `logsumexp` must be non-negative, including `g(W_h x_t)`. The original minGRU paper replaces `tanh` with the strictly-positive activation

```
g(x) = relu(x) + 0.5      # for x ≥ 0    → range [0.5, ∞)
g(x) = sigmoid(x)         # for x < 0    → range (0, 0.5)
```

The two branches are continuous at `x = 0` (both equal 0.5).

The **correct** `log_g` is:
- `x ≥ 0`: `log(x + 0.5)`
- `x < 0`: `log(sigmoid(x)) = -softplus(-x)`

These are also continuous at `x = 0`: both equal `log(0.5) ≈ -0.693`.

**What the code computes.** The negative branch is `5 - softplus(-x)`, off by exactly `+5` from the correct value. At `x = 0⁻` it returns `5 - log 2 ≈ 4.31`, jumping by ~5 nats discontinuously from the positive branch. After exponentiation:

| x      | true `g(x)` | parallel's effective `g(x)` | ratio |
|--------|-------------|------------------------------|-------|
| -3.0   | 0.0474      | 7.04                         | 148×  |
| -1.0   | 0.2689      | 39.91                        | 148×  |
| -0.1   | 0.4750      | 70.55                        | 148×  |
| 0.0    | 0.5000      | 0.50                         | 1×    |
| +1.0   | 1.5000      | 1.50                         | 1×    |

(Confirmed numerically — see verification section below.)

**Why this destroys the cell dynamics.** Whenever any unit of `W_h(x) < 0`, that unit's contribution to `h_tilde` is multiplied by `e^5 ≈ 148`. Since `h_t = (1 - z) · h_{t-1} + z · h_tilde`, the recurrent state inherits exploded contributions in those dimensions. After many steps these compound through the running mixture.

**Empirical verification.** Running the parallel scan against an exactly-equivalent sequential implementation that uses the *same* (intended) activation `g`:

```
PARALLEL vs SEQUENTIAL (matching g activation):
  max abs diff:  6.93e+01
  mean abs diff: 2.17e+01
  parallel mean |h|:    2.24e+01
  sequential mean |h|:  7.22e-01
```

The parallel scan's hidden state magnitude is **~30× larger** than the correct dynamics on random inputs. They are not the same function.

**How this causes masked-region collapse.** The model has to converge despite this 148× blow-up, and the cheapest way to do that is to learn weights that **avoid the buggy branch entirely**: keep `W_h(x) ≥ 0`. If gradients push `W_h.bias` (and the input projection layers feeding `W_h`) so that all pre-activations are non-negative, the cell uses only `g(x) = x + 0.5`, an affine map. The cell becomes effectively **linear**:

```
h_t ≈ (1 - σ(k_t)) · h_{t-1} + σ(k_t) · (W_h x_t + 0.5)
```

A linear-ish forward cell:
1. Has very limited capacity to encode complex temporal patterns (no real nonlinearity).
2. Treats masked positions (`flux=0, flux_err=0`) as zero contributions to `W_h x_t`. Since meta and conv embeddings are constant across timesteps, masked positions look like "small constant updates" — the state drifts smoothly through them rather than carrying information.
3. Pulls all hidden states toward similar "averaged" values, especially across long mask blocks.

The flow head, conditioned on a near-degenerate hidden-state distribution, falls back to the marginal `p(flux)`, which is centered at 0 for normalized light curves. **That's the collapse.**

### 2. Architectural issue: frozen sources inside mask blocks (secondary contributor)

The new loss [src/lcgen/utils/loss.py:138-149](src/lcgen/utils/loss.py#L138-L149) re-routes sources so masked targets pull from the nearest unmasked predecessor / successor:

```
fwd_remap[i] = max{j ≤ i : mask[j] == 1}      # via cummax
bwd_remap[i] = min{j ≥ i : mask[j] == 1}      # via cummin
```

For a target `j` inside a mask block `[a, b]` and offset `k`, both source defaults `j-k` and `j+k` typically also fall inside `[a, b]` (whenever `k < block_width`). Then:

- `fwd_idx = a - 1` (last unmasked before block)
- `bwd_idx = b + 1` (first unmasked after block)

**For *all* targets in the block, `src_f` and `src_b` are identical.** Only `t_enc[j]` and `ferr_tgt[j]` vary across the block. Predictions inside the block are functions of `t_enc[j]` (and `ferr_tgt[j]`) alone, with the heavy-lifting features (`h_fwd`, `h_bwd`) frozen.

Whether this collapses to flat predictions depends on:
- How much the head/flow learned to use `t_enc`. With the log_g bug forcing the cell toward affine dynamics, it's plausible the model under-weighted `t_enc` because the hidden state was already a weak signal.
- The dataset prior on flux (which is ≈ 0 for typical detrended TESS light curves).

Block size matters: with `min_size=5, max_size=720` and `mask_portion=0.5` over ~18,000-step sectors, the average block from log-uniform sampling is ~150 timesteps. Many blocks will be much wider than typical `k` (median sampled k ≈ 30 from log-uniform 1..720), so the frozen-source regime is the *dominant* training regime for masked targets.

### 3. Activation asymmetry between sequential and parallel (compounding factor)

Even after fixing the `log_g` bug, sequential and parallel modes use *different activations* for `h_tilde`:

- Sequential `step` ([simple_min_gru.py:54-60](src/lcgen/models/simple_min_gru.py#L54-L60)): `h_tilde = tanh(W_h x_t)` (range [-1, 1], symmetric, smooth).
- Parallel `step_parallel` ([simple_min_gru.py:95-113](src/lcgen/models/simple_min_gru.py#L95-L113)): `h_tilde = g(W_h x_t)` (range (0, ∞), strictly positive, asymmetric).

This is a deliberate paper-mandated tradeoff (positivity is required for the log-domain scan to work). But it means:

- A model trained in sequential mode is **not** equivalent to one trained in parallel mode. The state spaces, gradient flows, and feature encodings differ.
- The user's recollection that sequential "didn't behave this way" is plausible — `tanh`-based dynamics are richer and might have learned to use `t_enc` more effectively, or simply have not been pushed into the affine regime by an upstream bug.

### 4. Time-encoder underweighting (suspected, unverified)

`time_scale` is a learnable scalar parameter initialized at 5.0. If during training it drifted toward 0, the time slice's contribution to head input would shrink, making predictions inside frozen-source mask blocks even flatter.

**To check:** `model.time_scale.item()` after loading the trained checkpoint.

### 5. Inference-vs-training conditioning mismatch (NOT a bug here, but worth noting)

In [simple_min_gru.py:465](src/lcgen/models/simple_min_gru.py#L465) the **internal** reconstruction path uses `meas_err_t = x[:, ti, 1]`, where `x` has already been mask-zeroed. So that internal path conditions the flow on `flux_err = 0` for masked positions, while training conditions on the *real* `flux_err`. **This is inconsistent, but it does not affect the user's plotting code**: [scripts/plot_reconstructions.py:244](scripts/plot_reconstructions.py#L244) uses the unzeroed `flux_err` directly, matching training. So plot-time predictions are not corrupted by this issue. Worth fixing eventually for consistency.

## Why training still "converged"

A linearized cell + frozen sources can still drive val loss down because:

1. **Most targets are unmasked.** Only ~50% of positions are masked; for those that are, the model just learns the dataset prior. For unmasked targets the close-neighbor source hidden state carries enough information for the flow to fit.
2. **Logarithm of constant variance is bounded.** Predicting the prior mean with prior variance is a finite NLL, not infinite. The model is at the bottom of "predict the marginal" attractor and has no gradient pulling it elsewhere if the cell can't represent richer dynamics.
3. **The user's val_loss = 1.11** is consistent with a flow that is decent on unmasked targets and uninformative on masked targets — it's an *average* over both regimes.

## Verification

Run from repo root:

```bash
python -c "
import torch, torch.nn.functional as F
def g(x): return torch.where(x >= 0, x+0.5, torch.sigmoid(x))
def log_g_bug(x): return torch.where(x >= 0, (F.relu(x)+0.5).log(), 5 - F.softplus(-x))
def log_g_fixed(x): return torch.where(x >= 0, (F.relu(x)+0.5).log(), -F.softplus(-x))

xs = torch.linspace(-3, 3, 7)
print('x         g(x)     buggy_exp(log_g)  fixed_exp(log_g)')
for x in xs:
    print(f'{x.item():+5.2f}  {g(x).item():.4f}    {torch.exp(log_g_bug(x)).item():12.4f}    {torch.exp(log_g_fixed(x)).item():.4f}')
"
```

Expected: the "fixed" column matches `g(x)` exactly; the "buggy" column matches only for `x ≥ 0` and is ~148× too large for `x < 0`.

## Recommendations

### Priority 1 — Fix `log_g`

Change [src/lcgen/models/simple_min_gru.py:40-42](src/lcgen/models/simple_min_gru.py#L40-L42) to:

```python
@staticmethod
def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
```

This is a one-character-class fix (drop the `5 - `). It makes parallel mode mathematically equivalent to the intended sequential dynamics with `g(x) = relu(x)+0.5 / sigmoid(x)` activation.

**Important:** this changes the model. Existing parallel-trained checkpoints are *not* compatible — they have learned to compensate for the bug by keeping `W_h(x) ≥ 0`. After the fix, retraining is required from scratch. Fortunately the model is small (~95K params) and converges in ~100 epochs.

### Priority 2 — Re-evaluate the masking architecture after the fix

Once the cell has its intended nonlinear dynamics back, the "frozen source inside mask blocks" issue (Suspect 2) may become much milder, because:
- Hidden states will encode richer per-position information.
- `h_fwd[a-1]` and `h_bwd[b+1]` will carry usable summaries of the data on each side.
- Combined with `t_enc[j]` interpolation, the flow can produce smoothly varying predictions across the block.

Train fresh with the fix and inspect reconstruction plots before adding more architectural complexity. If predictions inside large mask blocks are still flat:

- **Investigate `time_scale`**: if it learned a small value, consider clamping it ≥ 1 or removing the learnable scaling.
- **Check `t_enc` magnitude**: if the time-encoder MLP outputs are tiny, the head is starved of position info inside frozen-source regions. Could increase `num_time_enc_dims`, add residual connection, or use Fourier features.
- **Add a per-position learnable embedding for "distance from nearest unmasked"**, fed into the head only for masked targets, so the flow can interpolate across the block more expressively. (Architectural change, only do this if simpler fixes don't suffice.)

### Priority 3 — Sanity test parallel vs sequential equivalence

Add a one-time test that asserts `BiDirectionalMinGRU(mode='parallel')` and `BiDirectionalMinGRU(mode='sequential')` (with weights tied) produce hidden states within numerical tolerance for a random input. This would have caught the `log_g` bug immediately. Sketch:

```python
# tests/test_parallel_sequential_equivalence.py
def test_parallel_equals_sequential():
    model_par = BiDirectionalMinGRU(hidden_size=8, mode='parallel', use_flow=False)
    model_seq = BiDirectionalMinGRU(hidden_size=8, mode='sequential', use_flow=False)
    # tie weights
    model_seq.load_state_dict(model_par.state_dict())
    # NOTE: this requires sequential to use g(x), not tanh(x). Sequential code
    # will need updating to use minGRUCell.g instead of torch.tanh inside step()
    # if we want this equivalence to hold.
    x = torch.randn(2, 64, 2); t = torch.linspace(0, 1, 64).expand(2, -1)
    out_par = model_par(x, t, return_states=True)
    out_seq = model_seq(x, t, return_states=True)
    assert (out_par['h_fwd_tensor'] - out_seq['h_fwd_tensor']).abs().max() < 1e-5
```

(Note: the current sequential `step` uses `tanh`, not `g`. So parallel and sequential are *expected* to differ even with `log_g` fixed. Either accept that as design or update sequential to use `g` for true equivalence.)

### Priority 4 — Inference-time `flux_err` consistency

In [simple_min_gru.py:465](src/lcgen/models/simple_min_gru.py#L465), `meas_err_t = x[:, ti, 1]` is the post-mask (zeroed) flux_err. For the internal reconstruction path to match training, store the original (pre-mask) flux_err at line 320 before mask is applied:

```python
flux_err_unzeroed = x[..., 1].clone()  # store before masking
# ... (masking happens here as before) ...
# ... (later, in the reconstruction loop:)
meas_err_t = flux_err_unzeroed[:, ti]
```

Low priority — only matters if anyone calls `model.forward(...)` without `return_states=True` for masked sequences. The plotting script doesn't.

## Summary of files to change

| File | Change | Purpose |
|---|---|---|
| `src/lcgen/models/simple_min_gru.py:40-42` | Drop the `5 - ` constant in `log_g` negative branch | Fix log_g bug (P1) |
| `src/lcgen/models/simple_min_gru.py:465` | Use unmasked flux_err in internal reconstruction | Consistency (P4) |
| `tests/` | Add parallel-vs-sequential equivalence test | Regression guard (P3) |

After the fix, retrain from scratch (do not resume — existing checkpoints have compensated for the bug) and re-plot reconstructions. If masked regions still show flat predictions, proceed to Priority 2 architectural investigations.
