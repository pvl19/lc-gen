# Sector-confound-robust age inference — applicability to non-cluster stars

**Date:** 2026-05-22
**Status:** IMPLEMENTED — three mitigations live in `scripts/kfold_age_inference.py`,
driven by `kfold_sector_robust.sh`, covered by `tests/test_sector_mitigations.py`
(8 tests). Tuning sweeps (especially the adversary λ) still TODO.
**Related:** [[project_sector_confound]], `docs/plans/2026-05-18_split-metadata-encoders.md`
(pretraining-stage attack on the same confound), `docs/plans/2026-05-18_metadata-masking-regularization.md`.

---

## 1. The problem

The age labels come almost entirely from **open clusters / co-eval populations**
(isochrones, gyrochronology). A cluster is, in practice, a `(sector, age)` pair:
its members fall in particular TESS sectors and share one age. So in the labeled
training set **sector predicts age at r ≈ 0.795** ([[project_sector_confound]]).

We want age inference that applies to **stars that are not part of the training
clusters** — field stars, but equally any star whose sector was not a labeled
cluster sector. For such a star, "sector X's cluster is age Y" is meaningless, so
a model that has learned the sector→age shortcut will systematically mispredict.

### How the confound enters the age-inference stage

The age flow conditions on `BPRP0, BPRP0_err, [MG]` — never on `sector`. But it
takes the **autoencoder latent**, which encodes sector at ~81 % top-1 (the probe
in `scripts/sector_partial_probe.py`). So the flow can read sector off the latent
and use it as an age shortcut even though sector is never an explicit feature.
This is **Route B** (light-curve-derived; the D7 ablation showed zeroing metadata
does not move the probe), so it cannot be removed by metadata changes alone — it
must be handled in the *representation use*, i.e. at the age-inference stage.

### What actually generalizes

Only the **within-sector** age signal transfers: age information beyond the
sector mean. That is exactly `sector_partial_probe` **test B (partial r)** —
regress the latent on `age − mean(age | sector)`. Maximising the model's use of
within-sector signal while minimising its use of the between-sector (sector-mean)
component is the whole objective. Always benchmark against the sector-only
baseline; only the latent's contribution *beyond* sector is real for new stars.

### Hard limit

The sector↔age coupling cannot be *removed* from a cluster-dominated set: a
sector containing only one age has no within-sector age variance to learn from.
The mitigations below **reduce reliance** on the shortcut and **measure**
generalization honestly; they do not manufacture signal that isn't there.

---

## 2. The three mitigations (all implemented)

All operate on **per-sample sectors**, so they require `--star_aggregation none`
(or `predict_mean`). Aggregated modes (`latent_max`, etc.) collapse sectors and
raise a clear error. `sectors` is threaded `run_kfold_cv → train_single_fold`
(7-tuple `TensorDataset` with a per-row sector index); a global
`sector → contiguous-index` map gives `n_sectors` for the adversary.

### Option 1 — Sector-disjoint cross-validation (diagnostic)

**Flag:** `--sector_level_split` (`--sector_split_keep_star_overlap` to disable
the star-overlap drop).

Holds out **whole sectors** per fold (shuffled sectors → folds); each fold
predicts stars from sectors absent in training — a direct proxy for "apply to a
star whose sector was never a training cluster." By default it also **drops val
rows whose star also appears in a training sector**, so the held-out set is
*star-disjoint as well as sector-disjoint* (a true unseen-star-in-unseen-sector
estimate). Dropped / unpredicted rows are left `NaN`; `plot_kfold_results` now
filters non-finite predictions before computing metrics, so normal modes (which
predict every row) are unaffected.

**What it tells you:** the gap between sector-disjoint MAE/r and a clean
star-disjoint baseline = how much accuracy was borrowed from the sector↔age
coupling. This is a *measurement*, not a fix — but you can't tune a fix without it.

### Option 2 — (sector × age) balanced sampling

**Flags:** `--balance_sector_age` (`--n_balance_age_bins`, default 10).

A `WeightedRandomSampler` weights each training row by `1 / count(sector, age-bin
cell)`, flattening the `(sector × age-bin)` joint. Over-represented cluster cells
(e.g. 500 same-sector-same-age stars) stop dominating the gradient, reducing the
*statistical incentive* to memorise sector→age. Closest to the user's original
intuition ("don't let it see lots of same-sector-same-age stars"). Limit: cannot
break the coupling for single-age sectors — there are no other-age stars in those
cells to up-weight.

### Option 3 — Gradient-reversal sector adversary (the strongest lever)

**Flags:** `--adv_sector_weight W` (peak λ, 0 = off; `--adv_hidden`, default 64).
Requires `--encoder_type mlp` (or `linear`), `prediction_mode='nle'`.

A DANN-style adversary (`GradReverse` autograd Function: identity forward,
gradient × −λ backward) sits between the **MLP bottleneck `z`** and a sector
classifier (`bottleneck → adv_hidden → n_sectors`). Minimising the classifier's
cross-entropy pushes the *encoder* to make `z` **sector-invariant** while the flow
still fits age. The adversary is added to the loss in encoder-training modes
(`'full'`, `'aux_only'`); skipped in `'nll_only'` (flow-only stage, encoder
frozen). λ is **ramped** `0 → W` over the epoch budget via the DANN sigmoid
`λ = W·(2/(1+e^{-10p}) − 1)`, `p = epochs_done / n_epochs` — a high λ from step 0
collapses the bottleneck before it learns anything. With `three_stage` training λ
keeps advancing through the flow-only stage (adversary inactive there), so it is
near peak by the stage-3 joint fine-tune. The adversary + GRL are **discarded at
inference**.

**Central risk (shared with Option 1):** sector correlates with *true* age, so an
over-strong adversary removes genuine age signal along with sector. Prior evidence:
in-fold sector-mean residualization drops the probe to ~7 % but **halves** the age
signal. So `W` must be **tuned, not maximised** — see §4.

---

## 3. Running it — `kfold_sector_robust.sh`

Hardcoded per-sector MLP+NF run (project convention: no args to `.sh`). Three
top-of-file toggles; `OUTPUT_DIR` auto-tags by the active set so runs do not
clobber (`sector-robust/none__baseline+secsplit+balance+adv0.3`, etc.).

```bash
SECTOR_SPLIT="false"        # option 1
BALANCE_SECTOR_AGE="false"  # option 2
ADV_SECTOR_WEIGHT="0.0"     # option 3 (peak λ)
```

Defaults: `STAR_AGGREGATION="none"`, `ENCODER_TYPE="mlp"`, ChronoFlow subset, the
3-stage MLP head matching the headline config, e110 cached per-sector latents.

---

## 4. Evaluation protocol

1. **Reference baseline** — `STAR_AGGREGATION="predict_mean"`, all toggles off:
   clean star-disjoint (no-leakage) MAE/r. (`none` + toggles off is sample-level
   and *leaks* multi-sector stars — optimistic upper bound only, not the honest
   reference.)
2. **Diagnose** — `none` + `SECTOR_SPLIT="true"`: sector-disjoint MAE/r. Gap vs
   (1) = sector-reliance.
3. **Mitigate + measure in one run** — keep `SECTOR_SPLIT="true"`, add
   `BALANCE_SECTOR_AGE` and/or sweep `ADV_SECTOR_WEIGHT` (e.g. 0.1 / 0.3 / 0.5 /
   1.0). The sector-disjoint r/MAE *with the mitigation on* is the tuning target:
   pick the largest λ that **closes the generalization gap without hurting
   in-distribution MAE**.

**Metric note:** for the **adversary**, the guardrail is the in-pipeline
**sector-disjoint generalization** number, *not* the `sector_partial_probe`
partial-r. That probe runs on the fixed 1536-d input latents, which the adversary
does not touch — it reshapes the learned MLP bottleneck, which is not currently
exported. (Exporting the bottleneck to probe it directly is a possible
enhancement — see §6.) The partial-r probe remains the right tool for the
*pooling-grid* comparison, which is about the input latents.

---

## 5. Caveats

- **Per-sector regime ≠ the headline.** These run with `--star_aggregation none`
  (per-sector rows), a different setup from the best `latent_max` (per-star)
  model. Compare sector-disjoint numbers against a per-sector star-split baseline,
  **not** the 0.912 / 0.139-dex headline.
- **Over-removal.** Options 1 (interpretation) and 3 (mechanism) can both erode
  real age signal because sector ⟂ age is false. λ needs a sweep.
- **Cost.** Each run is 300 epochs × 10 folds on CPU. Drop `N_EPOCHS` for quick
  λ sweeps.

---

## 6. Open questions / next steps

- **λ sweep** for the adversary; report the Pareto curve (sector-disjoint r vs
  in-distribution MAE).
- **Export the MLP bottleneck** during/after k-fold so the adversary's effect on
  the *learned* representation can be probed directly (sector top-1 on `z`),
  giving a representation-level Pareto axis to complement the generalization MAE.
- **Combine with pretraining-stage attacks**: the split-meta + adversarial-head
  model (e27, `2026-05-18_split-metadata-encoders.md`) attacks the confound in the
  *latent*; these attack its *use* downstream. They are complementary — measure
  whether stacking helps.
- **Per-sector light-curve normalisation** (the other Route-B option, not pursued
  here) as an input-side alternative to the representation-side adversary.
- Once a config wins, retrain the deployment `full_model` with it and re-validate
  on any available non-cluster age benchmark.
