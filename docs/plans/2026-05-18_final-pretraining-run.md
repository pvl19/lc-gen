# Final major pretraining run — consolidated change plan

**Date:** 2026-05-18
**Branch:** `age-inference-regularization`
**Status:** DRAFT — implementation blocked only on the metadata-masking decisions
D1–D7 (§3.1). Changes 1 and 2 are fully specified.

> **Intent.** This is meant to be the *final* major change to pretraining. The
> goal of this document is a complete, no-surprises inventory: every file that
> changes, every checkpoint-compatibility consequence, every coupled
> hyperparameter. If a change is not listed here, it is not in this run.

Related plan:
- `docs/plans/2026-05-18_metadata-masking-regularization.md` — Change 3 detail
  (DOROTHY-style metadata masking). Its decisions D1–D7 are still open.

---

## 0. Scope statement

The ONLY changes to pretraining in this run are the three below. Current
training code is committed (working tree shows only `.pyc`/`checkpoints/resume`
churn). The recent multiscale-pool change is inference-side only and does not
touch pretraining. Pretraining delta is exactly:

1. **Recurrence gating** of masked & padded positions (Change 1).
2. **Block-mask size & horizon:** `max_size = K = 2880` (Change 2).
3. **DOROTHY-style metadata masking** (Change 3).

Everything else — model architecture, loss math, optimizer, data — stays as in
the current `slurm_bridges2.sh`.

**Explicitly dropped from earlier drafts:**
- Input mask channel + learnable mask token — *superseded by gating* (§1.3).
- Separate "gap augmentation" (delete-block) — *redundant*; gating + keep-in-loss
  already reproduces a real gap for the encoder and adds inpainting (§1.4).

---

## 1. Change 1 — Recurrence gating for masked & padded positions

### 1.1 Problem
Currently the block mask is applied as `x = x * mask_expanded`
([simple_min_gru.py:338-344](src/lcgen/models/simple_min_gru.py#L338-L344)),
zeroing flux/flux_err at masked positions, but the RNN scan still **steps over
them**. In `parallel` mode (the trained mode) the scan is **not mask-gated** —
the header comment at [simple_min_gru.py:415](src/lcgen/models/simple_min_gru.py#L415)
("mask-gated updates") is inaccurate; no gating happens. Consequences:
- Masked positions perturb the hidden state (uninformative input, not absent).
- Padded positions leak into `h_bwd` of real positions: the backward scan starts
  at the last index (padding) and accumulates over padding before reaching real
  data.
- Masked flux≈0 is indistinguishable from genuine near-zero flux.

### 1.2 Fix — hard recurrence gating
Force the minGRU update gate to **`z = 0` at every `mask = 0` position** (block-
masked *and* padded), in both the forward and backward scans. Then `h_t =
h_{t-1}` — the masked step passes the state through unchanged and contributes
nothing to the recurrence.

Gating must be **hard/deterministic** (a `z := 0` override), not learned. Only a
hard override gives the exact "this position does not exist for the encoder"
behavior.

### 1.3 Why this supersedes the mask channel + learnable token
A hard-gated step has `h_t = h_{t-1}` and `h̃_t` fully discarded — the *entire
input* at that step (flux, flux_err, any mask channel, any learnable token, time
encoding) has **zero effect on the encoder** and receives **zero gradient**. A
learnable mask token at masked positions would never train (dead weight); a mask
channel would be constant `1` everywhere not gated (no information). The
channel/token design and gating are **substitutes** — gating is the stronger
one, and it removes the flux≈0 ambiguity directly (the encoder never sees masked
positions). Therefore: **no input mask channel, no learnable token.** Model
input stays `(B, L, 2) = [flux, flux_err]`.

### 1.4 Relationship to "gap augmentation" (resolved)
Because the time encoding is **absolute** (`t − t0`,
[simple_min_gru.py:354-360](src/lcgen/models/simple_min_gru.py#L354-L360)), the
real steps surrounding a gated block see a `(g+1)·cadence` jump in encoded time
— exactly what a real data gap (Q1: a time jump) produces. So for the *encoder*,
gating a masked block ≡ deleting it. Keeping the masked positions as loss
targets adds an **inpainting** objective on top. Hence:

> mask-gating + keep-in-loss = gap augmentation (encoder) + inpainting.

A separate delete-block gap augmentation is redundant and is **not** in scope.

### 1.5 Exact code changes
| File | Location | Change |
|---|---|---|
| `simple_min_gru.py` | `minGRUCell.step_parallel` | accept a per-step mask; override the update gate `z := 0` where `mask == 0`. Log-domain mechanism (the scan uses `cumsum`/`logcumsumexp`) to be determined by reading the cell — **must be verified, not assumed** |
| `simple_min_gru.py` | `minGRUCell.step` (sequential) | same gating, for mode parity |
| `simple_min_gru.py` | `forward` `~417-458` | pass `mask` to `forward_cell` scan; pass **flipped** `mask` to the `backward_cell` scan (the backward path flips the sequence — the mask must flip identically) |
| `simple_min_gru.py` | `:415` | fix/extend the stale "mask-gated updates" comment to describe the real behavior |
| `simple_min_gru.py` | `:338-344` | the `x = x * mask_expanded` zeroing is now redundant (gated out); may be removed for clarity or left (harmless) |

No constructor flag, no `rnn_input_dim` change, no CLI flag required — gating is
automatic whenever `mask` has zeros. `mask=None` (or all-ones) → no gating.

### 1.6 Loss interaction — VERIFY, do not assume
- Padding is excluded from the loss via `times == 0`, **independently of
  `mask`** ([loss.py:67-68,197-201](src/lcgen/utils/loss.py#L197-L201)). Block-
  masked targets are kept. So "gate every `mask=0` position, keep block-masked
  targets in the loss" is well-defined: gate padding + block-masks in the
  recurrence; the loss still drops padded targets (via `times`) and keeps
  block-masked ones.
- The loss already reroutes sources to the nearest unmasked position (cummax/
  cummin, [loss.py:139-155](src/lcgen/utils/loss.py#L139-L155)). Under gating,
  `h_fwd`/`h_bwd` at a masked position *already equal* the nearest-unmasked
  value, so the rerouting becomes **redundant**. It should compose harmlessly
  (idempotent gather), but this **must be confirmed** during implementation.
- The loss math itself needs no change.

### 1.7 Call sites — no change, but verify `mask` is passed
With no input channel, all 10 `torch.stack([flux, flux_err])` sites stay
`(B, L, 2)`. They only need to pass `mask=` (or `None`) so gating is controlled.
At inference `mask` is all-ones → gating is a no-op. Audited sites:
`train_simple_rnn.py:84,469`; `plot_umap_latent.py:571`; `predict_ages.py:170`;
`plot_reconstructions.py:210`; `plot_recon.py:101`;
`kfold_age_inference.py:715,1246`; `baseline_comparison.py:483`;
`visualize_time_enc.py:73`; `verify_plot_recon_consistency.py:37`.

---

## 2. Change 2 — Block-mask size & horizon K

### 2.1 Q1/Q2 findings
- **Q1:** real TESS gaps are *time jumps* — absent rows; adjacent array indices
  carry a large `dt`.
- **Q2 (measured, 40 sampled s77 curves/file):** per-curve max gap ≈ **9.95 days**
  (min = median = 9.95 d — one shared observing gap), up to **20.6 days** for
  stars spanning extra sectors. `time` is in days, 2-min cadence confirmed.

### 2.2 Units correction — the gap does NOT set `max_size`/`K`
A real gap occupies **one index transition** (`dt` up to ~20 d between adjacent
samples), not thousands of array positions. `max_size` and `K` are in
index/cadence-step units. Converting "10 days" → "~7,200 cadences" assumes the
gap is *filled* with samples — it is not. The original Change-2 rationale
("reflect the maximum gap") rests on a time-vs-index conflation and is dropped.
Real-gap robustness is handled by Change 1 (gating reproduces a gap) + the
absolute time encoder — not by `max_size`/`K`.

### 2.3 Decision (Q3 — resolved)
`max_size` = K = curriculum knob, chosen deliberately. **`--max_size 2880`,
`--K 2880`** (4 days of contiguous maskable samples). K tracks `max_size` so the
horizon can span a fully-masked block. Keep `--min_size 5`, `--mask_portion 0.5`.

Cost note: the loss samples a single `k` per batch, so larger K is ~cost-neutral
per batch; but log-uniform `k ∈ [1, K]` shifts probability toward large `k`,
asking for harder long-range predictions and diluting short-horizon training.
2880 is a deliberate, accepted trade.

---

## 3. Change 3 — DOROTHY-style metadata masking

See `docs/plans/2026-05-18_metadata-masking-regularization.md` for the full
design (concept mapping, `DynamicMetadataMasking` skeleton, adaptations).

### 3.1 Status — D1–D7 RESOLVED (one residual: `p_block`)
Resolved 2026-05-18 (detail in the metadata-masking plan §7):
- **D1:** goal of this run = encoder **robustness**; sector-confound *removal*
  deferred to later surgery.
- **D2:** **all 13 fields** maskable.
- **D3:** field masking = DOROTHY-style (`p_keep ~ U(0.3,1.0)`, guaranteed-keeper);
  whole-encoder block-drop = separate independent `Bernoulli(p_block)`.
  **Residual: `p_block` value** (recommend 0.10–0.20).
- **D4:** inference protocol **deferred** — it is a post-training choice, does
  not affect the encoder, does not block this run.
- **D5:** explicit binary mask channel — `MetadataEncoder(use_mask=True)`.
- **D6:** masking is a lightweight influence-reduction test; a deletion test may
  follow separately.
- **D7 — DONE.** Cheap-proxy sector-probe ablation run 2026-05-18: with metadata
  zeroed, the latent still encodes sector at ~82% top-1 (vs 82% with metadata) →
  the confound is **Route B (light-curve-derived)**. Change 3 masking will not
  move the sector probe; it stands as a robustness measure only. Sector-confound
  *removal* needs separate Route-B surgery (adversarial head / per-sector
  normalization). Detail: metadata-masking plan §6.8.

Change 3 is implementable once `p_block` is set. Implementation detail (the
`DynamicMetadataMasking` module, files) is in the metadata-masking plan §8.

> Note: the metadata `use_mask` channel (D5) is internal to `MetadataEncoder`
> and is **unrelated** to the (removed) input light-curve mask channel of old
> Change 1. Different component, different decision.

### 3.2 Metadata encoder size (Q8 — resolved)
Stays `[128,128,128] → 32`. Not enlarged: a 3×128 MLP for 13 inputs is not
capacity-bound, and enlarging it while adding a regularizer confounds
attribution. (Detail in the metadata-masking plan §3.2.)

---

## 4. Checkpoint compatibility

- **Change 1 (gating):** no weight-shape change → old checkpoints would *load*,
  but behave differently. Not a dimension break.
- **Change 2:** config only — no compatibility impact.
- **Change 3:** **iff D5 = metadata mask channel** (`use_mask=True`), the
  `MetadataEncoder` input dim goes 13 → 26 → hard dimension break. If D5 =
  zero-fill, no break.
- Regardless: this is a **fresh run**. `RESUME_FROM` in `slurm_bridges2.sh`
  (currently `checkpoints/resume/model.pt`) **MUST be cleared to `""`**. USER
  CONFIRMED (Q5).
- After the run: all cached latents
  (`final_model/parallel_fixed/e110_*/latents_*.npz`) and all age inference
  models trained on them are obsolete and must be regenerated/retrained.

---

## 5. Full file-change checklist

**Source:**
- [ ] `src/lcgen/models/simple_min_gru.py` — Change 1 (gating in
      `minGRUCell.step_parallel` + `.step`; pass mask + flipped mask from
      `forward`; fix `:415` comment); Change 3 (metadata masking hooks per D1–D7).
- [ ] `src/lcgen/train_simple_rnn.py` — Change 3 CLI flags + apply
      `DynamicMetadataMasking` in both train/val loops. (Change 1 needs no new
      flag; Change 2 is SLURM-only.)
- [ ] `src/lcgen/utils/` — new `DynamicMetadataMasking` (Change 3).
- [ ] `scripts/*` (10 call sites, §1.7) — verify each passes `mask`; no
      structural change. `plot_umap_latent.py` `load_model` only needs changes
      if D5 alters checkpoint contents.

**Loss:** no math change. **Verify** (§1.6) the rerouting/gating composition.

**Config (hardcoded per project convention):**
- [ ] `slurm_bridges2.sh` — `--max_size 2880`, `--K 2880`; clear `RESUME_FROM`;
      Change 3 flags once D1–D7 are set.
- [ ] `plot_umap.sh`, `predict_ages.sh` — mirror Change 3 inference protocol (D4).

**Docs:**
- [ ] `CLAUDE.md` — document recurrence gating, `max_size=K=2880`, the
      metadata-masking contract, obsolete-latents note.
- [ ] `final_model/final_parallel_e10/README.md` — new run row once trained.

---

## 6. Critical feedback

1. **`step_parallel` gating is the riskiest item.** It is real surgery on the
   log-domain parallel scan. Get it under a unit test before trusting it (§7).
2. **Change 2's original rationale was a units bug** (§2.2) — now corrected;
   `max_size`/`K` are decoupled from the gap.
3. **Three coupled changes, one run = weak attribution.** USER DECISION (Q5):
   accepted; separate runs not affordable. Recorded: if metrics move or regress,
   the cause will not be cleanly separable across Changes 1/2/3.
4. **Change 3 is fully specified except `p_block`.** D1–D7 resolved; set
   `p_block` (recommend 0.10–0.20) and implementation can proceed.
5. Per D1, this run targets **robustness**, not sector-confound removal. The D7
   ablation (now done) confirms the confound is **Route B** — so Change 3 will
   not move the sector probe, and the future removal effort must be a separate
   Route-B intervention (adversarial head / per-sector normalization), not more
   metadata work.

---

## 7. Validation plan

1. **Pre-flight ablation — DONE** (2026-05-18, metadata-masking plan §6.8):
   confound confirmed Route B. Post-run expectation set: the sector probe will
   *not* drop from Change 3.
2. **Gating unit test (mandatory before Bridges-2):** on a tiny local case,
   construct a sequence with a known masked block and assert the forward hidden
   state is *constant across the block* and that the post-block state equals
   what it would be with the block deleted. Same for the backward scan. Confirm
   masked-position inputs receive zero gradient.
3. **Smoke run:** short local CPU run (tiny `num_samples`) exercising gating +
   `DynamicMetadataMasking` + checkpoint save/load round-trip.
4. **Post-train:** sector linear/MLP probe on new latents (target drop from
   77–86% top-1); age k-fold MAE / Pearson r vs. current best (r=0.912,
   MAE=0.139 dex); sector-only baseline comparison.
5. Update `final_model/final_parallel_e10/README.md`.

---

## 8. Open items

- **`p_block`** — whole-encoder block-drop probability for Change 3 (recommend
  0.10–0.20). Only remaining unset parameter.
- **D4** inference protocol — deferred to age-inference time; does not block.
- Everything else is specified.
