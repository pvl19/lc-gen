---
date: 2026-04-29
status: APPROVED — open questions resolved 2026-04-29 (see §5)
---

# Combined pretrain + host k-fold age inference

## 1. Motivation

The current `kfold_age_inference.sh` trains and validates the age flow on labeled
stars from a single H5 (`final_pretrain/timeseries_pretrain.h5`). That set is
dominated by young cluster stars (median 79 Myr, range 1.5 Myr – 8 Gyr) but
sparsely populated above ~1 Gyr. Exoplanet host stars (NASA Exoplanet Archive
ages in `exop_hosts/archive_ages_default.csv`, ~2,430 stars) cover the older end
of the HR diagram more densely.

Goal: get a flow that generalizes to older stars by training on the *combined*
catalog, while still using cross-validation to honestly estimate held-out
performance on **hosts specifically**.

## 2. Proposed scheme

For each of `K` folds:

- Partition the **host stars only** into `K` equal-sized groups (by unique
  Gaia ID, so all sectors of one star stay in one group).
- Training set = (all non-host pretrain stars with valid ages) ∪ (host stars in
  groups other than `k`).
- Validation set = host stars in group `k`.
- Train a fresh flow + encoder; compute predictions for the held-out hosts.

Concatenate held-out predictions across folds → one prediction per host star,
each from a model that never saw it. Report MAE, Pearson r, etc., on hosts.

Optionally (`--train_full`): after k-fold, train one deployment model on all
labeled stars (hosts ∪ non-hosts).

## 3. Why this is different from the current pipeline

| Aspect | Current `kfold_age_inference.sh` | New `kfold_combined_age_inference.sh` |
|---|---|---|
| Train data | labeled stars in `--h5_path` only | non-host pretrain stars + (K-1)/K of hosts |
| Val data | held-out fold of `--h5_path` | held-out fold of hosts only |
| Fold structure | random K-fold over all labeled stars | K-fold over **host stars only**; non-hosts always in train |
| Reported metrics | over all stars in `--h5_path` | over hosts only |
| Latents extraction | single H5 | both H5s, latents kept separable by source |

## 4. Implementation strategy

### 4.1 Reuse `scripts/kfold_age_inference.py`

Rationale: the bulk of the script (latent extraction, encoder MLP, NSF flow,
training stages, posterior inference, plotting) is identical. The only changes
are (a) accepting two data sources, (b) constructing fold assignments differently,
and (c) restricting metrics to the held-out host fold.

### 4.2 Required changes to `scripts/kfold_age_inference.py`

1. **New CLI args**:
   - `--host_h5_path`           — second H5 file for host stars (optional; if
     absent, behavior is unchanged).
   - `--host_age_csv`           — separate CSV for host ages (optional; falls
     back to `--age_csv` if not given).
   - `--host_metadata_csv`      — CSV with host BPRP0 / BPRP0_err / MG (joined
     onto the age CSV by GaiaDR3_ID). Falls back to `--age_csv` if it already
     contains those columns.
   - `--combined_kfold`         — flag that switches to the new fold structure.
     Without it, the script behaves as today.
   - `--save_latents_pretrain` / `--load_latents_pretrain` — cache for the
     non-host source.
   - `--save_latents_hosts`    / `--load_latents_hosts`    — cache for the
     host source.

2. **Latent extraction**: when `--host_h5_path` is given, extract per-sector and
   cross-sector latents from both files **separately**, save each to its own
   cache if requested, then concatenate in memory tagged with
   `source ∈ {0, 1}` (0 = non-host pretrain, 1 = host). The tag rides through
   filtering and aggregation. Either side can be loaded from cache while the
   other is freshly extracted (e.g. recompute hosts only when their model
   path or H5 changes).

3. **Age loading**: extend `load_ages` to accept either a single CSV (current
   behavior) or two CSVs with a hint about which is the host source. For
   `archive_ages_default.csv`, normalize columns at load: `st_age * 1000 →
   age_Myr`, `st_ageerr1 * 1000 → age_Myr_err_hi`, `st_ageerr2 * 1000 →
   age_Myr_err_lo`. Photometry (BPRP0 etc.) joined from
   `--host_metadata_csv`.

4. **Fold assignment**: in `run_kfold_cv`, accept an optional `source` array.
   When provided AND `--combined_kfold` is set:
   - K-fold splitting runs on unique Gaia IDs of `source == 1` rows only.
   - All `source == 0` rows have `fold_of_sample = -1` (sentinel).
   - Inside the fold loop:
     - `train_idx = (source == 0) | ((source == 1) & (fold_of_sample != f))`
     - `val_idx   = (source == 1) & (fold_of_sample == f)`

5. **Metrics**: `plot_kfold_results` already operates on the predictions array
   indexed by `val_idx` per fold, so it naturally reports host-only metrics.
   Sanity-check that `kfold_predictions.csv` only contains host rows.

6. **`--train_full`**: train on `(source == 0) ∪ (source == 1)` — i.e. every
   labeled row from both files. No change beyond making sure the source array
   is plumbed through.

### 4.3 New shell wrapper: `kfold_combined_age_inference.sh`

Mirror `kfold_age_inference.sh` exactly but with:
- `H5_PATH=final_pretrain/timeseries_pretrain.h5` (non-host source)
- `HOST_H5_PATH=final_pretrain/timeseries_exop_hosts.h5`
- `AGE_CSV=final_pretrain/all_ages.csv` (non-host ages — same as today)
- `HOST_AGE_CSV=exop_hosts/archive_ages_default.csv`
- `HOST_METADATA_CSV=final_pretrain/host_all_metadata.csv`
- `SAVE_LATENTS_PRETRAIN=final_model/parallel_fixed/e60/latents_pretrain.npz`
- `SAVE_LATENTS_HOSTS=final_model/parallel_fixed/e60/latents_hosts.npz`
- `LOAD_LATENTS_PRETRAIN` / `LOAD_LATENTS_HOSTS` — empty by default; set
  either to skip extraction on that side.
- `--combined_kfold` flag added.
- `OUTPUT_DIR=final_model/parallel_fixed/3stage-10f-combined-${POOLING_MODE}-${STAR_AGGREGATION}`.

All other parameters (encoder type, pooling, training stages, etc.) inherit
the current best defaults from `kfold_age_inference.sh`.

## 5. Resolved decisions

- **Q1.** Host-age source = `exop_hosts/archive_ages_default.csv` (Gyr → Myr at
  load). Host BPRP0 / BPRP0_err / MG come from `exop_hosts/host_all_metadata.csv`
  (or `final_pretrain/host_all_metadata.csv` — same file, both copies exist).
  Stars missing BPRP0 are filtered out by the existing `valid_mask` logic.
- **Q2.** Validation = held-out hosts **only**. Non-host stars are train-only.
  `kfold_predictions.csv` will contain only host rows.
- **Q3.** `--train_full` trains on hosts ∪ non-hosts by default. Population
  reweighting is out of scope for v1.
- **Q4.** Confirmed: K-fold splits by unique Gaia ID over hosts; all sectors
  of a host stay in the same fold.
- **Q5. (changed)** Two **independent** latent caches, not one combined cache.
  Four CLI paths so each side can be saved or loaded independently:
  - `--save_latents_pretrain` / `--load_latents_pretrain`
  - `--save_latents_hosts`    / `--load_latents_hosts`
  - The script concatenates whatever it has at run time and tags each row's
    `source` (0 = pretrain, 1 = host) from which cache the row came from.
  - Existing `--save_latents` / `--load_latents` flags remain for the
    legacy single-source pipeline; the new script doesn't use them.
- **Q6.** Output dir = `final_model/parallel_fixed/3stage-10f-combined-${POOLING_MODE}-${STAR_AGGREGATION}/`.

## 6. Files to add / change

- **Add**: `kfold_combined_age_inference.sh` (new wrapper, ~150 lines)
- **Add**: `docs/plans/2026-04-29_combined-pretrain-host-kfold.md` (this file)
- **Modify**: `scripts/kfold_age_inference.py`
  - new CLI args (`--host_h5_path`, `--host_age_csv`, `--combined_kfold`)
  - `load_data_fresh` extended to accept and tag two sources
  - `run_kfold_cv` accepts optional `source` array and uses combined fold logic
  - latents-cache layout extended with `source` field (back-compat: missing →
    treat as single-source)
- **Modify**: `final_model/final_parallel_e10/README.md` — add a row for the
  new combined-training run once we have results.

## 7. Risks / things to watch

1. **Age scale mismatch**. `archive_ages_default.csv` is in Gyr. Need to
   normalize at load time to Myr, and convert error columns appropriately.
2. **Host metadata join**. If we use the archive CSV, BPRP0 / BPRP0_err must
   come from `host_all_metadata.csv` (or be re-queried). Stars missing
   BPRP0 will be filtered (matches existing `valid_mask` logic).
3. **Class imbalance**. Pretrain (~31k labeled rows) ≫ hosts (~2.4k stars
   × ~5 sectors = ~12k rows). Within a training batch the model will see
   mostly young stars. If this hurts host accuracy, consider reweighting or
   over-sampling hosts at training time. **Not** building this in v1; we'll
   measure first.
4. **Distribution shift between sources**. If host stars have systematically
   different stellar properties (e.g., metallicity priors, single-star bias),
   the flow may learn a source-dependent mapping. The current scheme will
   detect this via held-out host MAE — if it's worse than pretrain-only
   training would have been, that's a flag.
5. **Latents cache compatibility**. The two new caches
   (`latents_pretrain.npz`, `latents_hosts.npz`) live alongside the legacy
   `latents.npz` and are written/read by separate flags, so existing single-
   source workflows are untouched. The combined script errors clearly if
   `--combined_kfold` is set but neither host data nor a host cache is
   provided.

## 8. Validation criteria

The new pipeline is "successful" if:
- It runs to completion on the combined data without errors.
- Held-out host MAE is **lower** than the existing pretrain-only model's
  predictions on the same hosts (use `predict_ages.py` for the baseline).
- Pearson r on hosts improves correspondingly.

If host MAE is unchanged or worse, that's a meaningful negative result and
we'd document it in the README age-inference table.
