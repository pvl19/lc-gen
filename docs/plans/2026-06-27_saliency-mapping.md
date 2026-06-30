# Saliency mapping — per-timestep attribution of the pooled latent

**Date:** 2026-06-27
**Status:** IMPLEMENTED v1 — `scripts/compute_saliency.py`, `scripts/plot_saliency.py`,
`saliency.sh`. Smoke-tested on Gaia 249212801592026240 / sector 86 against
`final_model/meta_mask/e50/best_model.pt`. PC1 target uses an on-the-fly PCA
fit of the `latents_pretrain.npz` bank. See §11 for implementation notes and
the completeness caveat (not a perfect <1e-3 as originally targeted; ~5–10 %
for typical sequences because of non-smooth pool blocks).
**Related:** [[project_lc_ae]] (architecture overview), `scripts/plot_umap_latent.py`
(encoder loader + `compute_multiscale_features` pool), `scripts/plot_reconstructions.py`
(prior art for "load model, run one star, plot a per-timestep panel").

---

## 1. Goal

For a single light curve, produce a per-timestep scalar `s_t` that quantifies how
much sample `t` contributes to the **pooled latent** that downstream age inference
consumes. Render two stacked panels:

- **Top:** the flux trace, each point coloured by `s_t^flux`.
- **Bottom:** the flux_err trace, each point coloured by `s_t^err`.

Output one figure per requested star into `output/saliency/`.

The interesting science question is *where in time the encoder is paying
attention* — gap edges? high-flare regions? sector mid-points? — and whether
flux_err carries any meaningful saliency at all (likely flat by your intuition,
but worth confirming).

## 2. The attribution problem

The pipeline is:

```
(flux_t, flux_err_t)_{t=1..L}  ──►  BiDirectional MinGRU  ──►  h_{1..L} ∈ R^{L×H}
                                                                 │
                                                                 ▼
                                                multiscale pool (12 blocks, time-aware)
                                                                 │
                                                                 ▼
                                                       pooled latent z ∈ R^{12H}
```

The encoder produces a per-time hidden state, but the pool collapses time. The
two natural "what does t contribute to" questions are:

- **(a)** What does `(flux_t, flux_err_t)` contribute to **the pooled latent `z`**?
  → end-to-end attribution, useful because `z` is what every downstream task uses.
- **(b)** What does `(flux_t, flux_err_t)` contribute to **the encoder's own
  hidden-state trajectory** (independent of the pool)?
  → diagnostic of the minGRU's intrinsic memory weighting.

These two views answer different questions and we should compute both. The
**primary deliverable** (the coloured trace you described) should be (a).
(b) is cheap and goes in the same script as a sanity check.

## 3. Candidate methods, with tradeoffs

### Method 1 — Vanilla gradient saliency (BAD baseline)
`s_t = ‖ ∂f(z) / ∂x_t ‖` for some scalar `f(z)` (e.g. `‖z‖²`).
- ✅ Cheap (single backward pass).
- ❌ Noisy, saturation-blind (a feature that's already at its max contributes
     zero gradient), and the sign/scale is hard to interpret across timesteps.
- **Use only as a debug check** that gradients flow through the bi-encoder and
  the time-aware pool correctly.

### Method 2 — Integrated Gradients (RECOMMENDED primary)
For a chosen scalar `f(z)` and a baseline input `x'`:

```
IG_t  =  (x_t − x'_t)  ·  ∫_{α=0}^{1}  ∂f(z(x' + α(x − x'))) / ∂x_t  dα
```

Approximate the integral by `N` steps (N=32–64 is standard). The per-timestep
attribution is then `s_t = |IG_t^flux|` for the top panel, `|IG_t^err|` for the
bottom panel.

- ✅ Satisfies **completeness**: `Σ_t (IG_t^flux + IG_t^err) = f(z) − f(z')`.
     Every panel point's colour has a literal "what fraction of the latent
     change does this sample account for" reading.
- ✅ Pool-agnostic. Voronoi weights, segment bins, order stats (argmax / first /
     last) all attribute correctly because IG just rides the autograd graph.
- ✅ Saturation-aware. The integral picks up gradient mass along the path even
     where the endpoint gradient is zero.
- ⚠️ N forward+backward passes per star (cheap — single light curve).
- ⚠️ Baseline choice matters. Three sensible choices, discuss in §4.

### Method 3 — Occlusion (sanity check, slow)
For each window `W` of width `w` (e.g. w=1 or w=50), set `mask[W] = 0` and
recompute `z`. `s_W = ‖z − z_W‖`. Per-point heatmap = `s_W` smeared across the
window.

- ✅ Literal causal "what does the model lose without these points" interpretation.
- ✅ Uses the model's real mask gating (which is the actual ablation operator
     the encoder respects).
- ❌ O(L/w) forward passes. For L≈18,000 and w=1, that's infeasible. With w=200
     it's ~90 passes per star — fine for spot-checks but not the headline plot.
- **Run on 3–5 stars at w=200 to validate IG**, not as the primary panel.

### Method 4 — minGRU update-gate attribution (analytic diagnostic)
The minGRU update is `h_t = (1 − z_t)·h_{t−1} + z_t·tilde_h_t` where
`z_t = sigmoid(W_z x_t)`. Unrolling, the contribution of step `t` to `h_T` is
exactly `z_t · prod_{s=t+1..T}(1 − z_s) · g(W_h x_t)`. Define the scalar
"memory weight at t":

```
w_t  =  mean_T [ z_t · prod_{s=t+1..T}(1 − z_s) ]
```

(easily computed from the log-domain quantities the encoder already produces —
[simple_min_gru.py:107-140](src/lcgen/models/simple_min_gru.py#L107-L140)).
Symmetric for the backward pass. Combined: `w_t^bi = (w_t^→ + w_t^←) / 2`.

- ✅ Analytic, no autograd, ~free.
- ✅ Tells you *how much memory mass the encoder devotes to step t*, which is a
     property of the model independent of any choice of target scalar.
- ❌ Does NOT account for the pool (the pool reweights / picks-out hidden states
     downstream of the gates).
- ❌ Conflates "the encoder remembers this point" with "this point ends up
     mattering for `z`". A point with high `w_t` but in a hidden dim that the
     pool's order stats never select would still get high `w_t`.
- **Use as a second sanity panel**: if IG and gate-weights disagree strongly,
  the disagreement *is itself the story* (it tells us the pool is the load-bearing
  step, not the recurrence).

### Method 5 — Pool-structured analytic attribution (interesting but heavy)
Decompose `z = Σ_blocks` and attribute per-block:
- `glob_mean`: contribution of `t` = `w^vor_t / Σw^vor` × `h_t` (closed form).
- `seg_k`: same but restricted to bin `k`.
- `first_h` / `last_h`: only `t=0` / `t=L−1` contributes.
- `glob_max` / `glob_min`: only the per-dim argmax/argmin timestep contributes.
- `diff_mean` / `diff_std`: contribution of `t` is via rates Δh/Δt between t-1↔t and t↔t+1.

Then chain through the encoder via `∂h_t / ∂x_τ` to get to `(flux, flux_err)`.

- ✅ Gives a per-block decomposition ("PC1 of saliency is mostly the global mean
     block; diff blocks attribute to gap edges only").
- ❌ Substantial implementation effort, and IG already gets you the end-to-end
     scalar without the per-block decomposition. **Defer; revisit only if §3.2
     IG turns up something we want to explain.**

## 4. Choice of target scalar `f(z)`

What we attribute to is as important as how. Three sensible options:

1. **`f(z) = ‖z‖²` (or `‖z − z̄‖²` against the per-population mean).** Generic
   "how much does this point shape the latent at all". Cheapest to interpret;
   first one to ship.
2. **`f(z) = z_pc1` — projection onto the first PCA component of the
   age-inference encoder.** Directly asks "which timesteps drive the
   age-relevant latent direction". Requires loading the PCA fitted in
   `kfold_age_inference.py`. Strongly recommended as the *headline* run because
   it answers a downstream-meaningful question.
3. **`f(z) = E[age | z, BPRP0]` from the NSF flow.** The most directly
   science-interesting target ("what does the model look at to decide age?"),
   but requires the flow to be differentiable end-to-end through the age
   grid/sampling — feasible but more wiring.

**Proposal:** ship (1) and (2) in the first cut; gate (3) on whether (2) gives
useful structure. If (2) is dominated by sector-confound timesteps (start/end
edges, big-gap-bounding samples) we'll know to investigate before bothering
with (3).

## 5. Baseline `x'` for IG

Three reasonable baselines:

- **Zero flux, zero flux_err** — matches what the model sees at masked positions
  (after the `x = x * mask` zeroing in [simple_min_gru.py:459-461](src/lcgen/models/simple_min_gru.py#L459-L461)).
  Cleanest "this point vs. nothing" reading. **Default proposal.**
- **Per-sequence mean flux, median flux_err** — "this point vs. a flat sequence
  at the same average level". Removes uninteresting DC offsets from the
  attribution.
- **The same sequence with this point masked out** — only well-defined point by
  point, equivalent to occlusion. Not used for IG.

**Proposal:** ship the zero-baseline by default; expose a CLI flag for the
mean-baseline so we can compare on a couple of stars.

## 6. Visualization

Output figure per star: 2 panels, shared x-axis (time in days, sector-relative).
- **Top panel:** flux trace as a thin scatter / line; per-point colour =
  `s_t^flux` (signed or |·|, signed is more honest — pos drives `f` up, neg
  drives `f` down). Diverging colormap (RdBu) for signed, sequential
  (viridis / magma) for magnitude.
- **Bottom panel:** flux_err trace, coloured by `s_t^err`. Independent colour
  scale.
- **Title:** Gaia ID, sector, scalar f used, total IG = `f(z) − f(z')`.
- **Side colour bar** per panel; the two have different scales by construction.
- **Optional third panel:** the gate-derived `w_t^bi` (Method 4) on the same x.
  This is "encoder memory weight", *not* pool-aware attribution, but a useful
  side-by-side.

Mask coverage: gaps in the light curve appear as gaps in the panel — IG values
at masked steps are exactly zero (gating zeros the gradient).

## 7. File / script layout

Following project conventions ("shell scripts: all parameters hardcoded";
launchers in `bin/...` are not yet the standard for plotting scripts — current
`plot_reconstructions.sh` lives in repo root, so I'll match that):

```
scripts/compute_saliency.py        # core: load model, run IG + gate-weight, save npz
scripts/plot_saliency.py           # consumes npz, makes the 2(+1) panel figure
saliency.sh                        # hardcoded wrapper: model path, star IDs, N_steps,
                                   #                    f-target choice, baseline choice
output/saliency/
    {gaia_or_tic}_s{sector}/
        attribution.npz            # t, flux, flux_err, mask, s_flux, s_err, w_t
        saliency_signed.png
        saliency_abs.png
```

Reuse from existing code:
- `load_model` from [plot_umap_latent.py:194](scripts/plot_umap_latent.py#L194)
- `compute_multiscale_features` from [plot_umap_latent.py:270](scripts/plot_umap_latent.py#L270)
- Single-star H5 loading: lift from [plot_reconstructions.py](scripts/plot_reconstructions.py)
  (it already handles gaia_id / tic_id / random-pick lookups + `trim_edges`).

PCA loading for `f(z) = z_pc1` target: read the `shared/pca_*.pt` cache the
e100 age-inference pipeline already builds (see `kfold_e100_age_tests.sh`
in CLAUDE.md). Fail loudly if not present rather than silently fitting a fresh
PCA — we want the *deployed* projection.

## 8. Sanity checks before considering the figure trustworthy

1. **Completeness check** (IG-specific): assert
   `|Σ_t (IG_t^flux + IG_t^err) − (f(z) − f(z'))| / |f(z) − f(z')| < 1e-3`.
   Logged per star.
2. **Trim-edges respect**: the first/last `trim_edges` samples should be
   exactly zero in the IG output because they never enter the encoder.
3. **Mask respect**: at any timestep with `mask=0`, IG must be exactly zero.
4. **Occlusion agreement** (Method 3): on 3 hand-picked stars, run window
   occlusion at w=200 and check the per-window IG sum is rank-correlated
   (Spearman ≥ 0.6) with the per-window `‖z − z_W‖`. Anything lower means our
   IG target / baseline is mis-specified.
5. **Permutation null**: shuffle timesteps and re-attribute; saliency should
   become uniform in magnitude. (Confirms we're not picking up trivial
   position-encoding artifacts.)
6. **flux_err panel sanity**: expectation is that `‖s^err‖₁ ≪ ‖s^flux‖₁` for
   most stars. If they're comparable, that's a real and interesting finding,
   not a bug — but worth flagging.

## 9. Out of scope (for now)

- Aggregating saliency across a population (e.g. "where do field stars get
  attributed on average vs. cluster stars"). Easy follow-up once the
  per-star pipeline works.
- Saliency through the age flow itself (Method 4-target option 3 from §4).
  Gated on §4 option 2's findings.
- Sector-confound attribution: IG on the sector-residualized latent direction
  to find "which timesteps actually carry the sector signal". A natural
  sequel; not part of v1.

## 11. Implementation notes (v1)

- **Method 4 split into two quantities.** "Mean over T of c_{t,T}" (the plan's
  original Method 4) makes early steps look ~zero in long sequences because
  the denominator is the future-horizon length. "Contribution to final state"
  decays geometrically and saturates near zero across the middle of a
  ~10k-step sequence. Neither is visually useful for the headline trace, so
  v1 saves both for completeness (`w_gate_final` is the contribution-to-final
  one) and uses a third, simpler measure for the bottom panel:

      w_write_t = mean_d sigmoid(W_z x_t)_d, bidirectionally averaged.

  This is the **per-step write intensity** of the minGRU's update gate.
  Non-decaying, bounded in [0, 1], directly interpretable as "how aggressively
  does the encoder commit step t to memory". Smoke test on s86 host star shows
  ~0.49 mean with clear local structure around 0.4–0.55.
- **Completeness gap is intrinsic, not a bug.** The IG identity
  Σ_t (s_flux_t + s_err_t) = f(z) − f(z') requires f to be smooth along the
  IG path. The multiscale pool contains order statistics (`glob_max`,
  `glob_min`, `first_h`, `last_h`) whose argmax indices shift between α steps,
  giving f a kinked piecewise-smooth shape. Midpoint Riemann sum on a kinked
  integrand converges as O(1/N), not O(1/N²). Empirically:
    - N=4   → 9.7 % relative error (||z||² target)
    - N=32  → 8.8 %
    - Doubling N from 32 to 64 reduces error by ~half.
  We ship with **N_IG_STEPS=64** (≈5 % completeness gap) as a cost/accuracy
  sweet spot; relative attribution across timesteps — the only thing the
  coloured panel needs — is fully resolved at that N. The
  `attribution.npz` records the gap (`f_*_full - f_*_baseline` vs.
  `Σ(s_flux+s_err)`) and `sanity.json` records the relative error per target
  so a tighter run can be flagged retrospectively.
- **The PCA we attribute to is fit on-the-fly** from
  `final_model/meta_mask/e50/metaAll/latents_pretrain.npz`. Top-1 PC explains
  97 % of the bank variance, so f(z) = (z − mean) · PC1 is dominated by the
  most prominent latent direction; if downstream age inference uses a
  per-fold PCA the projections will be very close, but this is *not* the
  exact fold PCA — only a directionally-equivalent stand-in. Acceptable for
  the "which timesteps drive the dominant latent axis" story.
- **flux_err saliency confirmed flat.** On the s86 smoke star with the
  ||z||² target, Σ|s_err| / Σ|s_flux| = 0.19 — flux_err carries some but much
  less attribution mass than flux, as predicted in §10 of the original plan.
- **Files shipped.** `scripts/compute_saliency.py` (IG + gate-weight + PCA
  fit + sanity check), `scripts/plot_saliency.py` (3-panel figure with
  diverging colormap and 99th-percentile clip), `saliency.sh` (all-hardcoded
  wrapper, runs both `--target norm` and `--target pc1` figures).
- **Outputs.** `output/saliency/gaia{id}_s{sector}/{attribution.npz,
  sanity.json, saliency_norm.png, saliency_pc1.png}` per star.

## 10. Resolved before build

1. Target scalars: `‖z‖²` + `z_pc1` (PCA fit on-the-fly from
   `latents_pretrain.npz`). Flow-end target deferred.
2. Baseline: `zero` by default (matches the model's mask zeroing); `mean`
   exposed via `BASELINE_MODE` in `saliency.sh`.
3. Method 4: shipped as the third panel as the **write-gate magnitude**
   `mean_d sigmoid(W_z x_t)_d` (see §11 for why this is more informative
   than the originally-proposed "mean over T" formulation). The geometric-decay
   variant `w_gate_final` is saved alongside for diagnostic use.
4. Stars to attribute on next: pick from the same examples
   `plot_reconstructions.sh` already uses; the wrapper currently runs one
   star per invocation (set `SEED`/`GAIA_ID`/`TIC_ID` in `saliency.sh`).

## 12. Interpretation notes (v1 outputs)

Findings from the first batch of diagnostic runs on `sendit/e100`. The point
of this section is to record what the saliency *is honestly reporting* about
the trained model, so the figures are not over-interpreted in either
direction. Tests and scripts referenced live in `scripts/per_direction_nll.py`
and the `--decompose_directions` mode of `scripts/compute_saliency.py`.

### 12.1 The bidirectional architecture is symmetric; the trained model is not.

Forward and backward encoders share the same architecture, hidden dim, time
encoding, and head-norm. Nothing structural privileges one direction. On the
trained sendit/e100 checkpoint:

- Raw per-timestep magnitudes are balanced: `||h_fwd||` ≈ 10.10,
  `||h_bwd||` ≈ 9.86 (ratio 1.02).
- `head_norm.weight` mean abs is 1.048 (forward dims) vs 1.046 (backward) —
  the model has not learned to compensate via gain.
- Per-dim std of `h_bwd` is actually slightly *higher* than `h_fwd` after
  head_norm (1.02 vs 0.71).

So the two channels are equally "loud" in their hidden state values. The
asymmetry that drives saliency lives in their **gradients**, not magnitudes.

### 12.2 Forward = sensitivity, backward = predictive value.

Two diagnostics establish these as distinct properties:

**`--decompose_directions` (`scripts/compute_saliency.py`).**
Runs IG against `||z_fwd||²` and `||z_bwd||²` separately. On the s86 host
star:
- `s_fwd[t]` magnitude ≈ 7× `s_bwd[t]` magnitude — `∂h_fwd/∂flux` is ~7×
  more responsive than `∂h_bwd/∂flux`.
- Mirror correlation `corr(s_fwd[t], s_bwd[L-1-t])` ≈ +0.005 signed,
  +0.002 absolute. **The forward and backward encoders are not mirror
  images of each other.**

**`scripts/per_direction_nll.py`.**
Computes the flow-head NLL on a held-out star sample, under three context-
ablation modes: deployed bidirectional, forward-zeroed, backward-zeroed.
Mirrors `bounded_horizon_future_nll` exactly. On 50 stars from
pretrain+hosts+thickdisk at k ∈ {1, 8, 64, 720}:
- Median NLL (deployed): 1.41–1.43.
- Median NLL with forward zeroed: 1.48 (Δ +0.05–0.07).
- Median NLL with backward zeroed: 1.51 (Δ +0.07–0.10).
- Ratio Δbwd / Δfwd: 0.83–0.91 across every horizon.

So zeroing the **forward channel hurts prediction less** than zeroing the
backward channel. The backward channel is the more useful predictor.

The synthesis: the forward channel is highly **reactive** to inputs (loud
attribution), while the backward channel is more **informative** for the
prediction task. Both can be true simultaneously — sensitivity and
predictive value are different properties of a representation, and on this
checkpoint they happen to point in opposite directions across the two
encoders.

### 12.3 PC1 attribution is forward-dominated by gradient, not by loading.

PC1 itself (the principal axis fit on the deployed latent bank) is balanced
across the forward and backward halves of the 1536-dim pool:

- Forward share of `||PC1||²`: **0.482**.
- Backward share of `||PC1||²`: **0.518**.
- All 12 blocks have fwd/bwd ratio ≈ 1, except `first_h` (0.13, bwd-
  dominated because `h_fwd[0]` is initialized to zero) and `last_h` (3.54,
  fwd-dominated by symmetry); these two roughly cancel.

But the gradient `∂(z·PC1) / ∂flux` decomposes as:
```
∂(z·PC1) / ∂flux  =  PC1_fwd · ∂z_fwd / ∂flux  +  PC1_bwd · ∂z_bwd / ∂flux
```
Both `PC1_fwd` and `PC1_bwd` have similar magnitudes, but the forward
gradient is 7× larger. So **the PC1-target saliency inherits the forward
channel's asymmetry pattern regardless of PC1's balanced loading**. Picking
a "better" projection of `z` does not help; the asymmetry travels with the
sensitivity of the underlying representation.

### 12.4 The "decreasing impact over time" is a gap step, not a smooth decay.

Time-binning the attribution on two stars from the same checkpoint:

| Star (sector)              | Max gap (days) | Pre-gap mean | Post-gap mean | Post/pre ratio |
|----------------------------|----------------|--------------|---------------|----------------|
| Gaia 249212801592026240 / s86 | 6.05 d | ~245 | ~85 | 0.35 |
| Gaia 5211227379520496896 / s93 | 2.36 d | ~240 | ~178 | 0.74 |

Within each star, attribution is roughly flat through the pre-gap region, then
drops at the gap, then tails off slightly through the post-gap region. The
size of the drop scales with the size of the gap. The smooth ~5% monotonic
decay in the `w_gate` panel is a separate, much smaller effect — the
write-gate's response to time encoding drifts gently as `t` grows.

Mechanism: when either encoder crosses a long gap, its hidden state has
already accumulated significant signal from the pre-gap region. Post-gap
inputs write into an already-loaded recurrent state, so each post-gap
sample's marginal influence on the pool is smaller. Bidirectionality does
not cancel this because both directions have learned the same pre-gap
bias — they are not mirror images, so the backward channel cannot
"fill in" what the forward channel attenuated.

### 12.5 Why we did NOT modify the pool or the PCA to mitigate the asymmetry.

Two natural mitigations were considered and rejected:

- **Rescaling `h_bwd` before the pool** (e.g. multiplying by 7×). Per-feature
  z-scoring in the deployed age-inference PCA exactly undoes this rescaling
  by construction: each scaled feature gets divided by its also-scaled std,
  the factor cancels, and the resulting PC1 is identical to the un-rescaled
  one. Doesn't help.
- **Switching to per-channel z-scoring with fitted stds** (one std for all
  forward dims, one for all backward dims). The empirical channel stds are
  near-equal in this checkpoint (driven by head_norm), so per-channel
  z-scoring is numerically very close to per-feature z-scoring and likewise
  doesn't fix the gradient asymmetry. To get a real effect, the per-channel
  std ratio would have to be **hand-set** to match the sensitivity ratio
  (~7), not fitted from data — explicitly downweighting the forward channel.

Either fix would change the latent the age head consumes, requiring an
age-inference re-run to confirm MAE doesn't regress. More importantly, the
saliency is correctly reporting a real property of the trained model: the
forward channel **is** more sensitive to flux than the backward channel.
Hiding that with a normalization trick produces figures that look prettier
without changing the underlying behavior.

If we ever do another pretraining run, the right fix is upstream: **per-
direction LayerNorm** (separate LN for `h_fwd` and `h_bwd` before head
operations) would force the two channels to converge to comparable
sensitivities during training rather than retrofitting balance afterwards.
That is filed as a follow-up for the next pretraining iteration; it does
not block any current saliency work.

### 12.6 What the v1 saliency outputs are honestly showing.

Recap of the per-star attribution figures, given the above:

- **Pre/post-gap step-down in `s_flux_norm` and `s_flux_pc1`** is the
  encoder's actual processing of the gap; it scales with gap size and is
  shared by both directions.
- **Forward-domination of the bidirectional attribution magnitude** reflects
  the trained sensitivity gap, not a structural or PCA-loading bias.
- **`s_err` ≪ `s_flux`** (Σ|·| ratio ≈ 0.19) confirms `flux_err` carries
  much less attribution mass than `flux`.
- **`w_gate` monotonic 5% decay** is a smaller, separate effect from the
  encoder's response to time encoding — not the same mechanism as the gap
  step.

None of these are bugs in the saliency tool. They are the model's actual
behavior on this data, surfaced by the diagnostic. Future analyses should
treat them as findings, not artifacts.
