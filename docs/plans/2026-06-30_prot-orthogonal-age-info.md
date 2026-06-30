# Prot-orthogonal age information in the latent space

**Date:** 2026-06-30
**Status:** PLAN — scoped to ChronoFlow, scientific-understanding goal, cheap-first.
**Related:** `notebooks/age_inference_analysis.ipynb` (canonical latent-vs-gyro
LOSO comparison), `scripts/kfold_age_inference.py` (PCA → NSF age model),
`scripts/kfold_gyro_baseline.py` and `scripts/kfold_gyro_loso_inherit.py` (gyro
baseline), `scripts/kfold_latent_probe.py` (feature probes),
`output/latent_probes/sendit/e50/summary.md` (current probe table).

---

## 1. Question

On ChronoFlow cluster stars, latents (LOSO PCA4 r=0.732) beat the gyro
baseline (r≈0.50) by a wide margin. Some of that beat comes from Prot
information the latents have absorbed (probe r=0.92 for `tars_Prot`); the rest
is the question — **what age signal beyond Prot is the encoder using?**

Goal: a clear scientific answer, backed by a few diagnostic plots. Not a new
deployment model.

## 2. Three-step plan

```
  Step 1 — Quantify          Step 2 — Localize         Step 3 — Interpret
  ─────────────────          ─────────────────         ────────────────────
  How much age signal        Which directions in       What physical features
  remains after Prot is      the latent carry it?      do those directions
  accounted for?                                       correspond to?
```

Each step is gated on the previous one. If Step 1 finds the latent adds
essentially nothing over Prot on ChronoFlow, we stop. If it adds meaningful
signal, we move to Step 2 and pick the localization method based on what Step
1 told us. Step 3 reuses existing probe infrastructure on the Step 2 outputs.

## 3. Step 1 — Quantify (the next concrete deliverable)

Two pieces, both cheap, both run on the ChronoFlow LOSO setup:

**(a) Latents + Prot vs latents alone vs Prot alone.** Add a `--use_prot`
flag to `kfold_age_inference.py` that threads `log10(tars_Prot)` into the
NSF context alongside the 4 PCs. Run on the existing ChronoFlow LOSO folds.
Three numbers come out:

| Run | Inputs | r already known? |
|---|---|---|
| C1 | PCA4 only | yes, r=0.732 |
| C2 | Prot only | yes, r≈0.50 (gyro baseline) |
| C3 | PCA4 + Prot | **new** |

For an apples-to-apples comparison, restrict every row to the same stars
(those where `tars_Prot` is non-NaN). The cell of interest is
**∆r(C3 − C1)** — how much does giving the model Prot on top of the latents
improve things? If small, the latents already contain everything Prot knows
and the question "what beyond Prot?" reduces to "what's in the latents?". If
non-trivial, latents leak Prot information they can't reconstruct on their
own and the question is sharper.

**(b) Per-PC partial correlation.** For each of the top ~8 PCs of `z`,
compute `partial_r(PC_k, log10_age | log10(tars_Prot), BPRP0)`. This says,
per PC, "does this direction predict age beyond what Prot already explains?"
Cheap — a NumPy / Pandas calculation on the already-cached latents and ages.
Ranking PCs by this partial r tells us where the orthogonal signal lives, if
anywhere.

**Time budget for Step 1:** ~1 day total. Deliverable is one table (C1/C2/C3)
+ one ranking plot (PCs by partial r). After that we decide whether Step 2 is
worth doing and which Step 2 method fits.

## 4. Step 2 — Localize (sketch, not committed yet)

Three candidate methods, to be chosen *after* Step 1:

- **Project out the Prot direction.** Train a Prot regressor on `z`,
  subtract its predicted-Prot direction, run the age model on what's left.
  Best when Step 1(b) says the orthogonal signal is concentrated in a few
  PCs — projection then has a clear target.
- **Predict the gyro residual.** Run gyro to get an age prediction, take its
  residual to the truth, train a latent → residual model. Best when Step
  1(a) shows latents + Prot beats both alone — the residual is then exactly
  what the latents are picking up that Prot misses.
- **Find the age-predicting orthogonal subspace directly.** Constrain the
  age model to use only directions in `z` that are uncorrelated with Prot.
  Most principled, most engineering.

I'll bring concrete tradeoffs back when we have the Step 1 numbers in hand.

## 5. Step 3 — Interpret (sketch)

Whatever subspace Step 2 surfaces, re-run the existing latent probes
(`kfold_latent_probe.py`) using that subspace as input rather than full `z`.
Probe targets: existing ones (`flux_skew`, `flux_kurt`, `num_flares`,
`total_ed`, both Prot variants for sanity), plus new ones to be picked based
on what's interesting (candidates: BP−RP colour, RMS, ACF first-peak
amplitude as a spot-persistence proxy, low-frequency periodogram slope as a
granulation proxy). Probes whose r survives in the Prot-orthogonal subspace
are physical features the encoder represents *separately* from Prot, and are
candidate carriers of the orthogonal age signal.

This step also produces saliency difference maps for a few representative
stars (existing `scripts/compute_saliency.py` tooling), comparing where the
encoder looks for Prot vs where it looks for the Prot-orthogonal age target.

## 6. Decisions / open questions

These don't block Step 1, but are worth flagging now:

1. **`tars_Prot` is the default for Step 1.** It's the broader-coverage Prot
   variant and the one the latents recover most strongly. If the C3 − C1
   gap is suspiciously small, we re-run with `lit_Prot` as a sensitivity
   check to make sure we're not seeing a Prot-self-prediction artifact.
2. **PCA dim stays at 4.** Matches the existing ChronoFlow LOSO config and
   the canonical comparison in the notebook. Changing it mid-investigation
   muddles the comparison.
3. **Star aggregation.** ChronoFlow LOSO currently runs at `latent_max`
   per-star aggregation, which is what gives r=0.732. We stay there.
4. **Step 2 method.** Deliberately deferred until Step 1 numbers exist.

## Appendix A — full strategy menu (deferred but documented)

Six candidate strategies were considered during planning; three were chosen
above and three were deferred:

- **C (latents + Prot in NSF context)** → Step 1(a). Chosen.
- **D (per-PC partial correlation)** → Step 1(b). Chosen.
- **A (project out Prot)** → Step 2 candidate. Pros: literal `z⊥` plugs into
  existing pipeline. Cons: linear projection may leave nonlinear Prot residue;
  destructive of useful covariance.
- **B (predict gyro residual)** → Step 2 candidate. Pros: directly targets
  the question. Cons: conflates measurement de-noising of Prot with truly
  orthogonal info (relevant on field stars; less of an issue on ChronoFlow).
- **E (adversarial / GRL Prot-removal)** → **deferred**. Builds a Prot-blind
  deployment model, which is not the goal here.
- **F (matched-Prot pair analysis)** → **deferred**. Sample size on
  ChronoFlow is too small to support bin-conditional flow fits.

## Appendix B — reference numbers (sendit/e50, current production)

ChronoFlow:

| Source | r | MAE (dex) | N |
|---|---|---|---|
| latents PCA4 LOSO | 0.732 | 0.473 | 2470 |
| latents PCA4 LOCO | 0.709 | 0.457 | 2470 |
| gyro random | 0.503 | 0.536 | 2513 |
| gyro LOCO reference | 0.498 | 0.553 | 2513 |
| latents PCA8 random 10-fold (ceiling) | 0.828 | 0.278 | 2470 |

Probes (full `z`, not yet projected):

| Target | r (log10) |
|---|---|
| `tars_Prot` | 0.918 |
| `lit_Prot` | 0.610 |
| `num_flares` | 0.745 |
| `total_ed` | 0.863 |
| `flux_skew` | 0.991 |
| `flux_kurt` | 0.995 |
