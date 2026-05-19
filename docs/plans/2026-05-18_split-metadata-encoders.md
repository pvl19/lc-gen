# Split metadata encoders — instrumental metadata to the head, not the latent

**Date:** 2026-05-18
**Status:** PROPOSAL — experiment, post-final-run. Branch `split-meta-encoders`,
runtime-toggled by a `--split_meta_encoders` flag.
**Depends on:** the final pretraining run (`2026-05-18_final-pretraining-run.md`)
landing first; this is the next experiment, on whatever data is current then.

---

## 1. The proposal

Today all 13 metadata fields go through one `MetadataEncoder`, whose embedding is
broadcast to every RNN timestep (concatenated into the minGRU input). So the
embedding — including `sector/camera/ccd/cadence_s` — is baked into every hidden
state, hence into the pooled latent used for age inference.

Proposed split:
1. **Astrophysical metadata** (`G0, G0_err, BPRP0, BPRP0_err, parallax,
   parallax_error`, …) → keep in the current encoder, still injected into the
   RNN hidden states.
2. **Instrumental metadata** (`sector, camera, ccd, cadence_s`) → a *separate*
   encoder whose embedding conditions **only the prediction head**, never the
   RNN input. The model can still use it to reconstruct light curves, but it is
   not part of the hidden states, so pooling them should not carry it.

## 2. Reality check — what the D7 ablation already tells us

**This proposal targets Route A. D7 showed Route A is ≈ 0.**

D7 (metadata-masking plan §6.8): re-extracting `e110` latents with metadata
*fully zeroed* left the sector probe essentially unchanged — top-1 0.8215 →
0.8191 (chance 0.0725). Zeroing metadata is a close proxy for "instrumental
metadata not in the latent." So:

- **Expect the sector probe to stay ~82% after this change.** The confound is
  Route B — the light curve itself differs by sector (noise floor, cadence,
  systematics), and the RNN encodes that from the flux regardless of whether
  `sector` is also fed as metadata. Removing the metadata path cannot remove a
  signal the encoder reconstructs from the input.
- The proposal will **not, on its own, solve the stated problem.**

**What it *can* do** — and why it is still worth running:
- **Architectural hygiene.** Instrumental IDs genuinely should not sit in a
  latent intended for astrophysics; the split is correct by construction for the
  Route-A component, small as that is.
- **Mild indirect Route-B relief (plausible, unproven).** Giving the head direct
  access to `sector` lets it model sector-specific systematics itself, which
  *reduces the gradient pressure* on the RNN to encode sector for the sake of
  reconstruction. This is the standard "feed the nuisance variable to the
  decoder so the encoder need not learn it" disentanglement pattern. It helps
  only to the extent the encoder encodes sector *to lower reconstruction loss*
  (incentivized), not to the extent sector is *passively* present because noisy
  light curves simply produce different hidden states (mechanical). The latter
  is likely the larger share — so expect the effect to be small.
- **D7 secondary result mildly favors it.** The metadata-zeroed latent showed
  *more* age-signal-beyond-sector (partial r 0.262 → 0.322; joint−sector Δr
  +0.000 → +0.025) — OOD-caveated, but it hints the metadata path was
  reinforcing the sector *shortcut*. Removing instrumental metadata from the
  latent is consistent with that and unlikely to hurt.

**Bottom line:** low-risk, mildly-positive hygiene change; **not** a fix for the
sector probe. To actually move the probe, pair it with §7.

## 3. Field grouping — RESOLVED 2026-05-18

| group | fields | count |
|---|---|---|
| **Stellar encoder** (→ RNN hidden states) | `Tmag, parallax, parallax_error, G0, G0_err, BPRP0, BPRP0_err, median_flux, iqr_half_flux` | 9 |
| **Instrumental encoder** (→ head only) | `sector, camera, ccd` | 3 |
| **Dropped** | `cadence_s` | 1 |

Rationale (user): the stellar encoder gets everything that varies *per star*;
the instrumental encoder is restricted to exactly the fields that would *bias
age inference* — `sector/camera/ccd`. `Tmag`, `median_flux`, `iqr_half_flux`
vary per star (not per sector), so they go stellar-side.

`cadence_s` is **dropped**: verified constant (120.0 s) across all 60,631 light
curves — a dead feature. If TESS cadence diversity (e.g. 20-s) is ever added to
the dataset, `cadence_s` becomes an observational nuisance and should join the
*instrumental* group, not the stellar one.

Net: the 13-field metadata vector becomes **9 stellar + 3 instrumental**.

## 4. Architecture changes

- **Model (`simple_min_gru.py`):**
  - Two `MetadataEncoder`s: `astro_meta_encoder` (9 stellar fields, 32-dim
    embedding → RNN input as today) and `instr_meta_encoder` (3 instrumental
    fields, **16-dim** embedding → head context only).
  - `rnn_input_dim` uses only the astro embedding dim (smaller than today).
  - Head input gains the instrumental embedding: `gauss_head` input dim and the
    flow `context_dim` both grow by `instr_emb_dim`. Decide whether the
    instrumental embedding is concatenated before or after `head_norm` (it is
    not a hidden/time quantity — concatenate *after* the LayerNorm block).
  - `forward` returns the instrumental embedding (so the loss can use it).
- **Loss (`bounded_horizon_future_nll`, `loss.py`):** new optional arg
  `instr_emb` (B, instr_emb_dim); broadcast to `n_targets` and concatenate into
  the head/flow context. This is a real signature change — the one non-trivial
  edit outside the model.
- **Training (`train_simple_rnn.py`):** split the metadata tensor into the two
  field groups; pass both to the model; pass `instr_emb` to the loss.
- **Inference:** latent extraction (`plot_umap_latent.py`) uses `return_states`
  and does not run the head — the instrumental encoder does not affect the
  latent, so latent extraction is unchanged in output. `predict_ages.py` /
  reconstructions *do* run the head and must supply instrumental metadata.

## 5. Interaction with the metadata masking just shipped

The final run's `DynamicMetadataMasking` + `use_mask` channel assume ONE metadata
encoder over 13 fields. With the split (9 stellar + 3 instrumental, `cadence_s`
dropped):
- Astro encoder: keep DOROTHY-style masking (robustness) — unchanged rationale.
- Instrumental encoder: **left unmasked** (Q-B). It is out of the latent by
  construction, and the reconstruction head should reliably see sector — no
  `use_mask` channel, no `DynamicMetadataMasking` on this encoder.
- `DynamicMetadataMasking` + `meta_use_mask` apply to the **stellar encoder
  only** (now 9 fields). Mechanical change to the field-group plumbing.

## 6. Branch vs. flag — recommendation

It is **not** purely a batch argument: it changes module count, head/flow
context dims, and the loss signature — code, not config. But it *should* be
runtime-toggleable so the split model and the single-encoder baseline share one
codebase and can be ablated head-to-head.

**Recommendation:** develop on branch `split-meta-encoders`; gate the dual path
behind `--split_meta_encoders` (single-encoder = current behaviour when off).
Store the flag in the checkpoint; `load_model` reads it (as with `meta_use_mask`).

## 7. Adversarial sector head — flag `--adversarial_sector_head`

Per §2, the split alone will not move the sector probe (it targets Route A,
≈0). The adversarial head is the component that attacks Route B — the
light-curve-derived sector signal — by *penalising the latent for being
sector-predictable*. The two are complementary: the split gives sector a
legitimate home (the reconstruction head) so the model is not forced to smuggle
it into the latent; the adversary actively punishes whatever still leaks.

### 7.1 Mechanism — gradient reversal (DANN, Ganin & Lempitsky 2015)

A small classifier ("adversary") tries to predict `sector` from the pooled
latent. A **Gradient Reversal Layer (GRL)** sits between the latent and the
adversary: identity on the forward pass, gradient ×(−λ) on the backward pass.

```
hidden states ──pool──> latent ──GRL──> adversary MLP ──> sector logits ──> CE
```

```python
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)              # identity forward
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None   # reversed, scaled
```

Total training loss: `L = L_nll + L_adv`, where `L_adv` is cross-entropy of the
adversary's sector prediction. Backprop then does a **minimax**:
- **Adversary MLP params** sit *after* the GRL → receive the normal gradient →
  learn to predict sector as well as they can from the current latent.
- **Encoder params** sit *before* the GRL → the `L_adv` gradient reaching them is
  negated → they are pushed to make the pooled latent *less* sector-predictable,
  while still minimising `L_nll`.

At inference the adversary + GRL are discarded; only the (hopefully
sector-reduced) encoder remains.

### 7.2 Components & decisions

- **Pooling for the adversary.** The adversary must see something close to what
  age inference consumes (the multiscale latent). Running the full time-aware
  multiscale pool inside the training loop is expensive; recommend a cheap
  differentiable **masked mean+std of `h_fwd` and `h_bwd`** (4·hidden_size dims).
  Caveat: this scrubs the `glob_mean/std` blocks of the multiscale vector well
  but not directly its order-statistic blocks (`glob_min/max`) — a partial proxy.
- **Adversary MLP:** small, e.g. `4·hidden → 128 → n_sectors`; ~99 sector classes.
- **λ schedule:** ramp λ from 0 to `--adv_lambda_max` over training (DANN uses
  `λ = 2/(1+e^{-γp}) − 1`, p = training progress). A high λ from step 0 tends to
  collapse the latent before the encoder has learned anything — ramp it.
- **Sector labels:** the adversary needs integer sector *class* IDs, not the
  standardized `sector/100` float the metadata encoder gets. Thread the raw
  `sector` int through the dataset/collate (or recover it) and build a
  sector→index map for the CE target.
- **Train-only**, like `DynamicMetadataMasking`.

### 7.3 The central risk — sector correlates with age

This is not a free win. Sector *is* correlated with age (that is the whole
confound — open clusters of a given age fall in particular sectors). An
over-strong adversary removes everything sector-correlated, which **includes
genuine age signal**. Prior evidence: in-fold sector-mean residualization drops
the sector probe to ~7% but **halves** the age signal (`project_sector_confound`
memory). The adversary has the same failure mode.

So `--adv_lambda_max` must be *tuned*, not maximised. The objective is a Pareto
move: sector probe **down** while the within-sector age residual `r_residual`
(`sector_partial_probe.py` test B) stays **up**. Sweep λ; the right operating
point is the largest λ that has not yet started eroding `r_residual`.

### 7.4 Flags (this branch)

| flag | effect |
|---|---|
| `--split_meta_encoders` | route `sector/camera/ccd` to a separate head-only encoder |
| `--adversarial_sector_head` | enable the GRL adversary on the pooled latent |
| `--adv_lambda_max` (float) | peak GRL coefficient; ramped 0→max over training |

All independent → ablation grid: baseline / split / split+adversary / adversary-only.
Per-sector light-curve normalisation is the other Route-B option not pursued here
(see metadata-masking plan §6.4).

## 8. Cost & compatibility

- New architecture → checkpoint-incompatible with the final run → this needs its
  **own full pretraining run** on Bridges-2. Not a cheap experiment.
- All the §4 inference scripts need the split-aware model build.
- Run it as a clean A/B: `--split_meta_encoders` off vs on (vs on +
  `--adversarial_sector_head`), same data, same schedule, then compare the
  sector probe + age k-fold.

## 9. Decisions — RESOLVED 2026-05-18

- **Q-A** (§3): 9 stellar + 3 instrumental; `cadence_s` dropped.
- **Q-B:** instrumental encoder is **left unmasked** — it is out of the latent
  by construction, and the reconstruction head should reliably see sector.
  `DynamicMetadataMasking` applies to the stellar encoder only.
- **Q-C:** adversarial sector head **included**, behind its own CLI flag
  `--adversarial_sector_head` (independent of `--split_meta_encoders`). See §7.
- **Q-D:** instrumental embedding dim = **16**.

Remaining tuning (not blockers — set during the experiment): `--adv_lambda_max`
(λ sweep, §7.3).
