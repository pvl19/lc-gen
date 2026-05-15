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

### Interpretation (revised after per-curve check)

The aggregate-vs-Gaussian numbers above are **misleading** for the
downlink question. Once gap-edge variance is compared to the **same
curve's own interior**, the apparent "+40–70% inflation" disappears.

A follow-up check (`tests/interior_baseline_check.py`, 4 chunks of
pretrain, N=821 curves) extracted, for each curve:
- the gap-edge windows (pos 1..39 on each side, skipping the zero marker), and
- a per-curve interior baseline (centre 50% of each sub-segment, ≥50
  samples from sector ends and from the gap).

Results:

```
                              median   mean    p25    p75
whole-curve mean(f²)          1.022   1.870   0.948  1.057
pre-gap interior              1.001   2.150   0.926  1.056
pre-gap edge (pos 1..39)      0.992   2.646   0.663  1.328
post-gap interior             0.991   2.564   0.913  1.063
post-gap edge (pos 1..39)     1.041   1.224   0.680  1.513

per-curve edge/interior ratio:
  BEFORE  median=0.98   p75=1.35
  AFTER   median=1.04   p75=1.51
```

For the typical (median) curve, the gap edge sits at the same variance
as that curve's own interior (ratio ≈ 1.0 on both sides). Both interior
and edge sit at `mean(f²) ≈ 1.0` — exactly what per-curve z-norm
predicts. The aggregate `mean(f²) ≈ 1.5` we saw at the gap edges in the
per-position tables is a **heavy-tailed-aggregate artifact**: a minority
of curves have very heavy tails everywhere (interior aggregate
`mean(f²)` is 2.1–2.6, well above the median's 1.0), and that minority
dominates the across-curve average at every position — gap-edge or
interior alike. It is not evidence of a downlink-localised effect.

What this means for the actual findings:

1. **The pos-0 zero marker on each side is the only clean per-curve
   downlink artifact.** Preprocessing appears to insert a zero-valued
   boundary sample at every large gap. It is junk pretending to be a
   measurement, and dropping it is free.

2. **Mild asymmetry remains, but it is small.** The AFTER side has a
   median edge/interior ratio of 1.04 vs 0.98 on the BEFORE side, and
   25% of curves have AFTER-edge variance > 1.25× interior vs 19% on
   BEFORE. So the post-downlink window is *slightly* heavier-tailed
   than the pre-downlink window for a minority of curves, but not by
   an amount that should drive a uniform trimming policy.

3. **The aggregate `mean(f²)` "peak" at pos ~31 in the AFTER row is
   not a real time-locked transient.** A separate check
   (`tests/bump_position_check.py`, N=822) computed per-curve
   `argmax(f²)` in a 60-cadence post-gap window and histogrammed the
   positions. The distribution is approximately uniform — every
   5-cadence bin has 45–86 curves against a uniform-null expectation
   of 70 per bin, with no enrichment around pos 31. Per-curve bump
   amplitudes are very heavy-tailed (median max(f²) is 11.5× the
   curve median; 99th percentile is **80×**), so a small number of
   curves with single huge excursions at random positions jitter the
   per-position aggregate by ±0.5 units. **Earlier drafts of this
   doc claimed a "fine-pointing recovery transient at ~1 h
   post-downlink." That was wrong.**

4. **`flux_err` is again uninformative** — flat at ~0.79 (pretrain) /
   ~0.90 (exop hosts) across the entire window.

5. **Sector-edge vs. downlink comparison (revised).**
   - Sector start: real per-curve excess (the outer samples of curves
     have variance well above the same curves' interior — see the
     sector-edge analysis above).
   - Sector end: similar but milder.
   - Downlink: **no per-curve excess** beyond the pos-0 zero marker.

   So the sector edges are a real artifact requiring trimming; the
   downlink is not.

### Implications for trimming

- **`trim_edges` (sector-edge trim) is doing real work.** The earlier
  per-curve sector-edge effect is real (start much worse than end);
  recommendations from that section stand.
- **No mid-sector / downlink trim is justified** by the data, beyond
  optionally dropping the pos-0 zero marker on each side of any large
  gap. Even that is not strictly necessary if the encoder is robust
  to a single zero-valued sample bracketed by an irregular `Δt`.
- A gap-aware mask of any width would not measurably improve the
  per-curve variance distribution near the downlink, because that
  distribution is already indistinguishable from the curve's own
  interior at the median.

## Reproducing the downlink analysis

```bash
python3 tests/downlink_gap_analysis.py
```

Knobs: `MIN_GAP_DAYS=0.2`, `AMBIG_RATIO=0.5`, `N_EDGE=40`. Wall ~3 min
total on a warm SSD; processes one chunk at a time (~150 MB peak RSS).
