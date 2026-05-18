# Metadata-encoder masking for sector-leakage regularization

**Date:** 2026-05-18
**Branch:** `age-inference-regularization`
**Status:** DRAFT — open decisions unresolved (§7). Do NOT implement until D1–D7 resolved.

> **Provenance.** The source strategy is DOROTHY's *dynamic hierarchical masking*,
> documented in `docs/from_josh.md`. DOROTHY is a separate project (multi-survey
> stellar spectroscopy, APOGEE/GALAH labels). This document adapts that pattern to
> the lc_ae light-curve autoencoder. The DOROTHY scheme does **not** transfer
> wholesale — see §4 for what ports and §5 for what does not.

---

## 1. Goal

Reduce the degree to which the pretrained autoencoder latent encodes the
**observational/instrumental sector confound** (project memory: `sector_id` alone
predicts age at r=0.795; latent probes recover sector at 77–86% top-1).

Lever: randomly mask the metadata encoder during pretraining (whole-encoder
"block" masking + per-field masking) so the recurrent latent cannot become
dependent on metadata — in particular on `sector` / `camera` / `ccd`.

## 2. Current state of lc_ae

| Component | Location | Detail |
|---|---|---|
| Metadata encoder | `simple_min_gru.py:153-168` | `MetadataEncoder`, 3×128 MLP → 32-dim embedding, `use_mask=False`, `meta_dropout=0.1` (standard `nn.Dropout`, *not* feature masking) |
| Metadata embedding use | `simple_min_gru.py:362-368` | computed once, broadcast to every RNN timestep, concatenated into per-step input |
| Field list (13) | `MetadataAgePredictor.py:22-36` | `cadence_s, Tmag, sector, camera, ccd, parallax, parallax_error, G0, G0_err, BPRP0, BPRP0_err, median_flux, iqr_half_flux` |
| Standardizer | `MetadataAgePredictor.py:~398` | per-field scaling; NaN → 0 |
| Mask infra (exists, unused in GRU path) | `MetadataEncoder(use_mask=...)`, `RandomFeatureMasking` | mask-channel concat + Poisson feature masking |
| Existing time-block masking | `TimeSeriesDataset.apply_block_mask` (`:444`), `mask_portion` | contiguous block masking of the light curve already exists |
| Training entry | `slurm_bridges2.sh:116` | `--use_metadata` ON for the active run |
| Latent extraction | `plot_umap_latent.py:496-507` | also passes metadata → leakage path live at inference |
| Training loop | `train_simple_rnn.py:71-88`, `456-477` | metadata moved to device, passed to `model(...)` |

## 3. DOROTHY → lc_ae concept mapping

DOROTHY has two augmentation classes, each a 2-level hierarchy. Mapping:

| DOROTHY construct | lc_ae analog | Status |
|---|---|---|
| `DynamicInputMasking` — **survey** level (drop a whole survey) | Drop the whole **metadata encoder** for a star ("block mask") | ✅ port this |
| `DynamicInputMasking` — **wavelength block** (contiguous log-uniform block) | Contiguous **time-block** masking of the light curve | ⚠️ already exists (`apply_block_mask`); out of scope here |
| `DynamicLabelMasking` — **labelset** level | (pretraining has no labels) | ❌ no analog |
| `DynamicLabelMasking` — **individual label** | Mask an **individual metadata field** | ✅ port this |

The two operations we want — whole-encoder block masking and per-field masking —
correspond to DOROTHY's **survey-level keep/drop** and **individual-label
masking**. The metadata tensor `(B, 13)` is structurally identical to DOROTHY's
`(batch, n_params)` tensor, so DOROTHY's 6-step vectorized pattern transfers
almost verbatim.

### 3a. DOROTHY's 6-step vectorized pattern (reference)
Per masking level, vectorized over the batch (no Python per-sample loop):

```
available  = mask.any(dim=-1)                 # (B, n_groups) — has real data
any_avail  = available.any(dim=1)             # (B,)
rand       = torch.rand(B, n_groups, ...)     # one draw per (sample, group)
# guaranteed keeper: argmax of rand with unavailable set to -inf
rg = rand.clone(); rg[~available] = -inf
guaranteed_idx = rg.argmax(dim=1)             # (B,)
keep = (rand < p_keep) & available            # independent Bernoulli
keep[torch.arange(B), guaranteed_idx] = True  # force-keep one per row
keep = keep | ~any_avail.unsqueeze(1)         # leave empty rows untouched
```

Per-batch the keep-probability is resampled `p_keep ~ U(p_min, p_max)`; all
samples in a batch share `p_keep` but draw independent Bernoullis.

## 4. What ports cleanly

- The **6-step vectorized pattern** — directly, treating metadata `(B, 13)` as one
  group of 13 elements.
- **Per-batch `p_keep` resampling** then per-sample independent Bernoulli draws.
- **Whole-encoder block masking** = DOROTHY survey-level: one Bernoulli per star
  over a single group.
- The **mixed numpy/torch RNG approach**: `np.random.uniform` for per-batch
  scalars, `torch.rand(device=...)` for the per-(sample,field) matrix.
- Applying masking **train-only, once per batch, after `.to(device)`, before the
  forward pass** (DOROTHY `trainer.py:1547`).

## 5. What does NOT port — required adaptations

### 5a. lc_ae has no metadata mask channel
DOROTHY's design rests on 3-channel tensors `[value, error, mask]` where the loss
multiplies by the mask channel — masking is "free." lc_ae metadata is a bare
`(B, 13)` value tensor with **no mask channel and no per-field uncertainty**.
Adaptation: enable `MetadataEncoder(use_mask=True)` (exists; concatenates a binary
mask, **doubles encoder input dim**) — see decision D5. Without it, masked fields
are ambiguous (the standardizer already maps NaN → 0, so "0" ≠ "masked").

### 5b. Metadata is an *input*, not a *label*
This is DOROTHY's `DynamicInputMasking` semantics, **not** `DynamicLabelMasking`.
DOROTHY's label masking is convenient because a masked label drops out of the
heteroscedastic loss. Masking a metadata *input field* only changes what the
encoder sees; `bounded_horizon_future_nll` is unaffected. Consequence: the "loss
already handles ignore-where-mask=0" shortcut does **not** apply here — this is
purely an input-side augmentation. (This is fine for the goal; just don't expect
loss-side simplicity.)

### 5c. Drop the "guaranteed keeper" for the field path
DOROTHY force-keeps ≥1 label per star (a fully-empty label tensor = no
supervision). For lc_ae metadata, "all 13 fields masked" is *exactly the
block-drop case* and is desirable. Keeping the guaranteed-keeper would make full
block-drop unreachable via the field path. Recommendation: **omit the
guaranteed-keeper** for metadata field masking; treat block-mask and field-mask
as the two independent knobs (D3).

### 5d. No "natural" availability mask to seed `available`
DOROTHY computes `available` from real missingness in channel 2. lc_ae would
construct it from the standardizer's NaN→0 record, or treat all 13 fields as
always-available. Decision D5 / implementation detail.

### 5e. Inference protocol
DOROTHY: *"masking is never applied at validation/test time."* If lc_ae copies
this, `sector` is fully present at inference → **the leakage is intact** (see
§6.1). Decision D4 must resolve this; it is the crux, and `from_josh.md` does not
address it because DOROTHY simulates missing surveys, it does not remove a
confound.

## 6. Critical feedback

### 6.1 Masking-as-augmentation ≠ leakage removal
Random masking teaches the model to *cope without* a field; it does not penalize
*using* `sector` when present. In the robustness regime (full metadata at
inference, DOROTHY-style) the model still fully exploits sector. The only config
that genuinely removes leakage masks the confound fields *at inference too* — at
which point the fields are effectively deleted, so the simplest honest
implementation is to **drop `sector, camera, ccd` (and likely `cadence_s`) from
`DEFAULT_METADATA_FIELDS`**. Probabilistic masking is strictly weaker than
deletion for the stated goal.

### 6.2 The leakage is Route B — CONFIRMED 2026-05-18
The latent has two routes to `sector`: **Route A** — the metadata encoder
ingests `sector` literally; **Route B** — the light curve itself differs by
sector (noise floor, cadence, scattered-light systematics, 27- vs 60-day
baselines; CLAUDE.md notes UMAP islands for sectors 97/98). Masking the metadata
encoder closes only Route A.

**Result of the D7 proxy ablation (§6.8):** the `e110` model's latents were
re-extracted with metadata zeroed (`--zero_metadata`) and probed for sector
identity. Sector top-1 accuracy was **essentially unchanged**: 0.8215 (full
metadata) → 0.8191 (metadata zeroed); chance 0.0725. So the latent encodes
sector at ~82% **via the light curve alone — Route B dominates.** This is the
trustworthy direction of the proxy (a *surviving* signal is conclusive).

**Consequence:** metadata masking (this whole plan) will **not** move the sector
probe. It remains justified as the D1 robustness measure, but sector-confound
*removal* requires Route-B surgery (see §6.4). Do not expect the post-run sector
probe to drop.

### 6.3 Field masking is blunt — separate confounds from physics
`parallax, G0, G0_err, BPRP0, BPRP0_err` are genuinely age-informative
(`BPRP0`/`BPRP0_err` are literally the age flow's conditioning variables).
Masking them during pretraining removes real signal. Restrict field masking to
confound fields (`sector, camera, ccd, cadence_s`, possibly `Tmag, median_flux,
iqr_half_flux`). A single global `p_field` over all 13 fields would be a mistake.

### 6.4 Removal requires a Route-B objective (now the *required* path)
With §6.2 showing the confound is Route B, sector-confound *removal* (D1's
deferred goal) **cannot** be done metadata-side. The removal surgery must target
the light-curve-derived sector encoding directly. Candidates:
- **Adversarial / gradient-reversal head** predicting `sector` from the latent,
  trained against — penalizes leakage regardless of route.
- **Per-sector light-curve normalization** before encoding (remove sector-level
  flux/noise offsets).
- Probing-and-residualizing sector-aligned latent directions.
This masking plan is the mild/cheap robustness option; it is **not** the removal
solution.

### 6.5 Cost and blast radius
Full pretraining rerun on Bridges-2. Enabling `use_mask=True` changes the meta
encoder input dim → existing checkpoints incompatible regardless. All cached
latents (`final_model/parallel_fixed/e110_*/latents_*.npz`) become obsolete;
every age inference model must be retrained — same blast radius as the recent
multiscale-pool change.

### 6.6 New train/inference-consistency contract
Whatever inference protocol is chosen (D4) becomes a hard contract mirrored in
`plot_umap.sh` and `predict_ages.sh`, exactly like `--trim_edges`. A mismatch
silently reintroduces leakage or creates OOD inputs.

## 7. Decisions — RESOLVED 2026-05-18 (one residual: `p_block`)

- **D1 — RESOLVED.** Primary goal of *this* run is **encoder robustness**.
  Sector-confound *removal* is wanted eventually but deferred to separate later
  surgery. So this masking is the robustness step; it is not expected, by
  itself, to remove the confound. (Reframes §6 — see §6.7 below.)
- **D2 — RESOLVED.** **All 13 fields** are maskable (robustness goal — the
  encoder should not depend on any single field). Note: masking
  `BPRP0/parallax/G0` does not harm age inference — the age flow conditions on
  `BPRP0` etc. *directly*, not via the latent, so the latent need not carry
  them. §6.3's "confound-only" caution applied to a leakage-removal goal, which
  is now D1-deferred — overridden.
- **D3 — RESOLVED (field level), one residual.** Field masking = DOROTHY-style
  verbatim: per-batch `p_keep ~ U(0.3, 1.0)`, per-sample independent Bernoulli,
  guaranteed-keeper retained. **Residual:** the whole-encoder block-drop cannot
  reuse DOROTHY's survey-level code — with a *single* group, DOROTHY's
  guaranteed-keeper argmax force-keeps it every time → it would never drop.
  Block-drop must be a **separate independent `Bernoulli(p_block)`**, and
  `p_block` has no DOROTHY default. **`p_block` value still needed** —
  recommend 0.10–0.20.
- **D4 — DEFERRED (correctly).** The inference protocol does not affect the
  encoder's training — it is a post-training choice at latent extraction. It
  does not block this run. Useful property: heavy training-time masking makes
  *both* "full metadata" and "masked-confounds" in-distribution at inference, so
  the choice stays genuinely free until age-inference time.
- **D5 — RESOLVED.** Explicit binary mask channel — `MetadataEncoder(use_mask=
  True)`. Meta-encoder input dim 13 → 26 → checkpoint dimension break (expected;
  fresh run).
- **D6 — RESOLVED.** This run is a lightweight test of *reducing* the influence
  of metadata fields, not removing them. A separate future test may *delete*
  `sector/camera/ccd` outright. Masking-vs-deletion is understood and accepted.
- **D7 — RESOLVED (method below).** The pre-flight sector-probe ablation will be
  run, but per D1 it is **no longer a go/no-go gate** for this run (the run is
  justified by robustness alone) — it is now an input to the *future* removal
  surgery. Feasibility with current artifacts: see §6.8.

### 6.7 Reframe note (D1)
With D1 = robustness, the §6.1/§6.4 critique ("masking ≠ leakage removal") is no
longer an objection — it is the *acknowledged* scope. Removal is a later effort.
This plan is correctly framed as the robustness step.

### 6.8 D7 cheap-proxy ablation — DONE 2026-05-18
A clean ablation would need a new `use_metadata=False` run (no such checkpoint
exists; all `final_model/parallel_fixed/*` are metadata-trained). The cheap proxy
was run instead: `e110` latents re-extracted with `--zero_metadata`
(`plot_umap_latent.py` new flag; `metadata` standardized then zeroed →
`meta_encoder(0)` = constant embedding → per-star latent variation is
light-curve-only). Saved to `final_model/parallel_fixed/e110_nometa/`.

`scripts/sector_partial_probe.py e110_timeaware e110_nometa` results (N=32171
labeled rows, 9287 stars, chance 0.0725):

| metric | e110_timeaware (full metadata) | e110_nometa (metadata zeroed) |
|---|---|---|
| sector probe top-1 | 0.8215 | 0.8191 |
| sector probe top-3 | 0.9230 | 0.9298 |
| age: latent-only r | 0.6548 | 0.7123 |
| age: joint−sector Δr | +0.0001 | +0.0249 |
| partial r (age beyond sector) | 0.2623 | 0.3221 |

**Conclusion (trustworthy — surviving signal):** sector identity stays
recoverable at ~82% top-1 with metadata fully withheld → the confound is
**Route B** (light-curve-derived). Metadata masking will not move it.

**Secondary (OOD-caveated, suggestive only):** the metadata-zeroed latent shows
*more* age signal beyond sector (partial r 0.262 → 0.322; joint−sector Δr
+0.000 → +0.025) — hints the metadata path was reinforcing the sector shortcut
rather than adding genuine age info. Not conclusive (the model is OOD when run
without metadata), but it is mild positive evidence for the robustness value of
masking metadata.

## 8. Implementation (decisions resolved — §7)

### 8a. New module: `DynamicMetadataMasking`
Location `src/lcgen/utils/`. Adapted from DOROTHY's `_apply_single` /
`_apply_hierarchical`. Operates on `metadata (B, 13)` + `meta_mask (B, 13)`
(D5 = mask channel), **train-only**:

1. **Block level** — per star, an **independent `Bernoulli(p_block)`** decides
   whether the whole metadata embedding is suppressed. NOT DOROTHY's survey-level
   code: with a single group, DOROTHY's guaranteed-keeper would force-keep it
   every time (§7 D3). Suppression = zero `meta_emb` for that row + zero its mask
   row (explicit "all-missing" signal).
2. **Field level** — DOROTHY-style verbatim over **all 13 fields** (D2): per-batch
   `p_keep ~ U(0.3, 1.0)`, per-(star,field) `rand`, `keep = rand < p_keep &
   available`, **guaranteed-keeper retained** (DOROTHY-style, D3). Masked field →
   value 0 + mask channel 0. (Full-suppression is still reachable via the
   independent block-drop in step 1, so the guaranteed-keeper here is fine.)
3. Block-masked stars skip the field step (already fully suppressed).

```python
class DynamicMetadataMasking:
    """Train-only metadata masking. Block level = independent Bernoulli(p_block);
    field level = DOROTHY-style hierarchical masking over all 13 fields."""
    def __init__(self, p_block, p_keep_min=0.3, p_keep_max=1.0):
        ...
    def __call__(self, metadata, meta_mask):
        # metadata: (B, 13); meta_mask: (B, 13)
        # returns (metadata_out, meta_mask_out, block_drop)  -- train-only
        ...
```

**Open value:** `p_block` (§7 D3 residual) — recommend 0.10–0.20.

### 8b. Files to change
- `src/lcgen/utils/` — new `DynamicMetadataMasking`.
- `src/lcgen/models/simple_min_gru.py` — `MetadataEncoder(use_mask=True)` (D5);
  accept a block-drop vector to zero `meta_emb` rows after `meta_encoder(...)`.
- `src/lcgen/train_simple_rnn.py` — instantiate the masker; apply in both
  training loops (`:71-88`, `:456-477`) after `.to(device)`; gate on `training`.
- New CLI args: `--meta_block_mask_prob`, `--meta_keep_min`, `--meta_keep_max`,
  `--meta_use_mask`.
- `scripts/plot_umap_latent.py` — `load_model` must handle the `use_mask=True`
  checkpoint (meta encoder input dim 26). Build/pass `meta_mask` (all-ones at
  inference unless D4 later says otherwise).
- `slurm_bridges2.sh` — hardcode the new flags.
- `plot_umap.sh`, `predict_ages.sh` — mirror the D4 inference protocol *once D4
  is decided* (deferred — until then, full unmasked metadata = all-ones mask).
- `CLAUDE.md` — document the flags + the (future) D4 consistency contract.

### 8c. RNG / DDP note
Per-batch scalars via `np.random.uniform`; per-(sample,field) matrix via
`torch.rand(device=...)`. Under DDP each rank masks its own shard independently
— acceptable and desirable. For strict reproducibility, plumb a `torch.Generator`
(DOROTHY itself does not, per `from_josh.md` §8).

## 9. Validation plan

1. **Pre-flight ablation (before any retraining):** sector probe on a
   `use_metadata=False` latent (§6.2). If sector still highly recoverable, stop
   and re-scope toward Route B / adversarial removal.
2. After retraining: re-run the sector linear/MLP probe on new latents; target a
   drop from 77–86% top-1 toward chance.
3. Age inference k-fold (`kfold_age_inference.sh`): MAE / Pearson r vs. current
   best (r=0.912, MAE=0.139 dex). Watch for regression.
4. Sector-only baseline comparison (project memory: always benchmark against it)
   — confirm the model-vs-sector-only gap widens.

## 10. Recommended sequencing

1. Run §9 step 1 ablation.
2. If Route A is confirmed material, resolve D1–D6.
3. Implement §8, retrain on Bridges-2, regenerate latents, retrain age models.
4. Validate (§9).
