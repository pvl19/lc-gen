# Edge-trim analysis — is `trim_edges=10` enough?

**Date:** 2026-05-15
**Scope:** statistical characterisation of leading/trailing-edge artifacts in
the pretraining H5 files, used to assess the current `trim_edges=10` default.

## Motivation

`trim_edges=10` was chosen as a naive buffer to strip TESS sector-edge
artifacts (scattered light at sector start, thermal settling at sector end)
before the encoder sees the data. This note quantifies how far the artifacts
actually extend, so the trim value is informed rather than guessed.

## Method

Per-curve flux is z-normalised at preprocessing time, so a clean interior
behaves like N(0,1). The interior baseline is therefore theoretical:

| stat | baseline (Gaussian) |
|---|---|
| `mean(f²)` | 1.000 |
| `median(\|f\|)` | 0.6745 |
| `P(\|f\|>3)` | 0.0027 |

Edge artifacts inflate all three.

For each H5 file we read 8 contiguous chunks (`chunks=(256, full_cols)`,
gzip-compressed) → ~2048 light curves per file, then computed per-position
aggregates over the first/last 40 samples of each curve. Chunk-aligned reads
were necessary because random row-indexed reads on the gzip-compressed
chunks decompressed ~30 MB per row and never finished.

Script: `tests/edge_trim_analysis.py` (mirrored to `/tmp` during the
investigation).

Files analysed:
- `final_pretrain/timeseries_pretrain.h5` (40,662 curves, median length 15,375)
- `final_pretrain/timeseries_exop_hosts.h5` (20,xxx curves, median length 16,419)

## Results

### `timeseries_pretrain.h5`

**START edge** (col 0 = outermost sample):

| pos | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 39 |
|---|---|---|---|---|---|---|---|---|---|
| `mean(f²)` | 3.89 | 2.40 | **2.12** | 2.02 | 1.71 | 1.49 | 1.44 | 2.62 | 2.20 |
| `median\|f\|` | 0.80 | 0.78 | **0.80** | 0.77 | 0.77 | 0.76 | 0.75 | 0.72 | 0.76 |
| `P(\|f\|>3)` | 0.040 | 0.031 | **0.030** | 0.026 | 0.024 | 0.016 | 0.018 | 0.017 | 0.018 |

First position within +X% of baseline `mean(f²)`: +50% at pos 24, +20%/+10%/+5% never within 40.

**END edge** (col 0 = outermost sample, i.e. last index of the curve):

| pos | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 39 |
|---|---|---|---|---|---|---|---|---|
| `mean(f²)` | 1.26 | 1.52 | **1.44** | 1.42 | 1.41 | 1.37 | 1.34 | 1.30 |
| `median\|f\|` | 0.59 | 0.73 | **0.75** | 0.73 | 0.77 | 0.75 | 0.75 | 0.76 |
| `P(\|f\|>3)` | 0.017 | 0.022 | **0.019** | 0.020 | 0.014 | 0.016 | 0.013 | 0.008 |

First position within +50% of baseline `mean(f²)`: pos 0; never within +20%
through pos 40.

### `timeseries_exop_hosts.h5`

Same qualitative shape. Slightly worse start edge:

| pos | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 39 |
|---|---|---|---|---|---|---|---|---|
| START `mean(f²)` | 3.02 | 2.33 | 3.05 | 1.94 | 1.58 | 1.63 | 1.88 | 1.54 |
| START `P(\|f\|>3)` | 0.055 | 0.042 | 0.036 | 0.031 | 0.024 | 0.022 | 0.026 | 0.020 |
| END `mean(f²)` | 1.51 | 2.12 | 1.73 | 1.54 | 1.47 | 1.27 | 1.32 | 1.22 |
| END `P(\|f\|>3)` | 0.020 | 0.023 | 0.015 | 0.018 | 0.013 | 0.010 | 0.011 | 0.012 |

Two single-curve outliers (start pos 7: `mean(f²)`=884; end pos 17:
`mean(f²)`=10.9) inflate isolated cells and should be ignored — the
median/`P(|f|>3)` columns are robust to them.

## Interpretation

1. **`trim_edges=10` clears the bulk of the artifact but not the tail.**
   By position 10 the median is back to ~Gaussian (`median|f|` within ~15%
   of baseline), but the variance is still ~2× interior at the start and
   ~1.5× at the end. The excess is driven by **outlier tails**: `P(|f|>3)`
   is still ~10× the Gaussian rate at position 10.

2. **Asymmetry is real.** At the same position offset, the start edge
   carries 2–3× the variance excess of the end edge. Scattered light at
   the start of a sector is the worse problem; trailing-edge thermal
   settling is mild by comparison.

3. **Secondary artifact ~30 samples in.** `mean(f²)` rebounds to 2.6–3.7
   at start positions 31–35 in `timeseries_pretrain.h5`. ~30 cadences ≈
   1 hour into the sector — likely a recurring instrumental feature
   (post-momentum-dump? post-fine-pointing?). A simple uniform trim
   wouldn't cleanly remove it; would need a per-sector mask informed by
   the TESS data-release notes if it matters downstream.

4. **`flux_err` is uninformative for edge detection.** It sits at
   ~0.86–0.89 across all positions in both edges, indistinguishable from
   interior. The artifact lives in the flux excursions, not the reported
   uncertainty.

## Recommendation

`trim_edges=10` is the rough floor — it cleans the median but leaves
heavy outlier tails the encoder still has to absorb. Defensible options:

- **Bump to `trim_edges=20`** (single-knob change). Cuts start-edge
  variance excess from ~2× to ~1.5× and outlier rate from ~3% to ~2%.
  Negligible data loss given median length ≈15k. Recommended if any
  bump is made.
- **Asymmetric trim (e.g. 25 leading, 10 trailing)** would be the
  cleanest match to the data, but requires changing
  `LazyH5Dataset` / `TimeSeriesDataset` to accept a tuple instead of a
  scalar. Worth doing if edge artifacts turn out to drive any
  observed UMAP or age-inference behaviour.
- **Status quo (`trim_edges=10`)** is fine if the encoder is observed
  to be robust to the residual edge inflation in downstream metrics.

No change is being made to the default in this commit — this note is
the evidence base for whichever choice the next training run adopts.
Any change to `trim_edges` invalidates the current cached latents and
checkpoints (see `CLAUDE.md` for the compatibility note).

## Reproducing the sector-edge analysis

```bash
python3 tests/edge_trim_analysis.py
```

Knobs at the top of the script: `N_CHUNKS`, `N_EDGE`, `SEED`. Reading
8 chunks per file (~2048 curves each) takes ~30 s on a warm SSD; the
gzip decompression dominates wall time. Do **not** switch to random
row indexing without a chunk-cache strategy — it makes the script
unrunnable.

---

## Mid-sector downlink-gap analysis (full population)

The same statistical machinery, applied to the trailing/leading edges of
the segments on either side of the mid-sector data downlink. Run on every
curve in both files (`tests/downlink_gap_analysis.py`); wall time was
129 s for `timeseries_pretrain.h5` and 44 s for `timeseries_exop_hosts.h5`.

### Method

For each curve:
1. Find the largest `Δt` in the valid region.
2. Require the gap to be > 0.2 days (TESS downlinks are ~1–4 d; normal
   cadence is ~0.0014 d) **and** unambiguous (second-largest gap must
   be < 50% of the largest).
3. Require ≥ 40 valid samples on each side of the gap.
4. Aggregate flux per position with col 0 = sample closest to the gap,
   on both the "BEFORE" (trailing edge of pre-gap segment, reversed)
   and "AFTER" (leading edge of post-gap segment) sides.

### Population stats

| file | total | kept | gap size median (d) | gap pos median (frac) |
|---|---|---|---|---|
| `timeseries_pretrain.h5` | 40,662 | 34,576 (85%) | 2.44 (p10–p90: 1.16–4.86) | 0.50 (tight) |
| `timeseries_exop_hosts.h5` | 19,969 | 12,568 (63%) | 2.08 (p10–p90: 0.93–4.88) | 0.50 (tight) |

The downlink sits almost exactly mid-sector for both files. Exop hosts
have a higher ambiguous-gap rate (37%), consistent with that population
having more multi-sector concatenations.

### Results — `timeseries_pretrain.h5` (N=34,576)

**BEFORE gap** (col 0 = closest to gap, looking backwards):

| pos | 0 | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 39 |
|---|---|---|---|---|---|---|---|---|---|---|
| `mean(f²)` | **0.66*** | 1.51 | 1.41 | 1.49 | 1.57 | 1.59 | 1.43 | 1.47 | 1.45 | 1.50 |
| `median\|f\|` | 0.43* | 0.72 | 0.70 | 0.70 | 0.70 | 0.69 | 0.69 | 0.70 | 0.70 | 0.71 |
| `P(\|f\|>3)` | 0.002* | 0.014 | 0.011 | 0.011 | 0.009 | 0.008 | 0.009 | 0.009 | 0.008 | 0.008 |

**AFTER gap** (col 0 = closest to gap, looking forward):

| pos | 0 | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 31 | 35 | 39 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mean(f²)` | **0.69*** | 1.53 | 1.43 | 1.49 | 1.72 | 1.63 | 1.71 | 2.27 | **2.28** | 1.93 | 1.89 |
| `median\|f\|` | 0.42* | 0.74 | 0.69 | 0.70 | 0.71 | 0.70 | 0.69 | 0.70 | 0.70 | 0.70 | 0.70 |
| `P(\|f\|>3)` | 0.002* | 0.013 | 0.011 | 0.010 | 0.010 | 0.010 | 0.009 | 0.008 | 0.010 | 0.008 | 0.009 |

`*` Position 0 is anomalously **low** (`mean(f²) ≈ 0.66`) on every cut.
This is a preprocessing artifact, not a physical signal — most likely
the boundary sample on each side of the gap is zero-imputed (z-normed
data with zeros pulls `mean(f²)` and `median|f|` toward zero). It is
**not** a sample that survived a real shutter exposure adjacent to the
downlink. Treat pos 0 as junk; the real artifact behaviour starts at
pos 1.

`timeseries_exop_hosts.h5` (N=12,568) is qualitatively the same shape;
post-gap `mean(f²)` plateaus higher at ~1.7 across all 40 positions
without a clean recovery within the window.

### Interpretation

1. **Downlink edges are not clean.** From position 1 onward, `mean(f²)`
   sits at **~1.4–1.7** on both sides — a persistent **+40–70% variance
   excess** that does not decay across the 40-cadence (~80 min) window we
   sampled. `median|f|` recovers to within ~5% of Gaussian by pos 1, so
   the inflation is again **outlier-tail-driven** (`P(|f|>3) ≈ 4–6× the
   Gaussian rate at pos 1, settling to ~3× by pos 30`).

2. **Asymmetric "settling bump" on the AFTER side.** `mean(f²)` rises
   from ~1.5 at pos 5 to a peak of **2.28 at pos ~31** in pretrain, then
   relaxes back. ~31 cadences ≈ 1 hour after the downlink — same
   timescale as the secondary bump we saw at the start of a sector. This
   is consistent with a fine-pointing / thermal recovery transient
   following the downlink slew, not with the immediate post-shutter-open
   sample.

3. **`flux_err` is again uninformative** — flat at ~0.79 (pretrain) /
   ~0.90 (exop hosts) across the entire window.

4. **Sector-edge vs. downlink comparison.**
   - Sector start: extreme spike at the outermost samples (`mean(f²) ≈ 4`),
     decaying to ~2 by pos 10, ~1.5 by pos 25.
   - Sector end: ~1.4 plateau, much milder.
   - Downlink (either side): ~1.5 plateau across the full 40-pos window
     (after the pos-0 imputation marker), with a delayed +30-cadence
     bump on the AFTER side.

   So the downlink behaves more like a "second sector end + second sector
   start" with the worst part being **a delayed transient**, not the
   sample immediately adjacent to the gap.

### Implications for `trim_edges`

`trim_edges` only operates at the outer ends of each curve — it does
**not** touch the downlink. The findings above mean the encoder is
seeing a mid-sector region with persistent +50% variance inflation on
both sides of the downlink, plus a +130% bump ~1 h after the gap on
the AFTER side. Cleaning that would require either:

- masking a window around the largest `Δt` per curve, or
- a gap-aware encoder that conditions on the irregular time axis (which
  the model already does, but only at the per-step time-encoding level —
  it does not currently mask out the settling transient).

Neither change is being made in this commit; the doc is the evidence
base for whichever route the next training run takes. If only one knob
is touched, **adding a downlink-aware mask of ~32 cadences on the AFTER
side** would have higher impact than changing `trim_edges` from 10 to 20.

## Reproducing the downlink analysis

```bash
python3 tests/downlink_gap_analysis.py
```

Knobs: `MIN_GAP_DAYS=0.2`, `AMBIG_RATIO=0.5`, `N_EDGE=40`. Wall ~3 min
total on a warm SSD; processes one chunk at a time (~150 MB peak RSS).
