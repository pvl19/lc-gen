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

## Reproducing

```bash
python3 tests/edge_trim_analysis.py
```

Knobs at the top of the script: `N_CHUNKS`, `N_EDGE`, `SEED`. Reading
8 chunks per file (~2048 curves each) takes ~30 s on a warm SSD; the
gzip decompression dominates wall time. Do **not** switch to random
row indexing without a chunk-cache strategy — it makes the script
unrunnable.
