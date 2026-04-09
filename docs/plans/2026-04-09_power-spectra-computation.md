# Power Spectra & ACF Computation

**Date:** 2026-04-09
**Status:** Implemented

## Context

The conv encoder branch of BiDirectionalMinGRU can ingest power spectra, f-statistics, and ACFs as auxiliary channels during pretraining, but these aren't stored in the H5 files yet. The original computation code was in `proj_lc_ages/analysis/process_lightcurves.ipynb` using `tapify.MultiTaper`. That code had several issues; this plan documents the fixes and the new clean implementation.

## How the multi-taper spectral estimation works

1. **tapify.MultiTaper** computes the power spectrum using Discrete Prolate Spheroidal Sequence (DPSS) tapers — orthogonal window functions designed to minimize spectral leakage.
2. Parameters: `NW=4` (time-bandwidth product), `K=7` (number of tapers). Resolution bandwidth = 2×NW/T_obs.
3. `adaptive_weighting=True` iteratively reweights tapers to minimize broadband bias.
4. `ftest=True` computes the F-statistic for harmonic line detection at each frequency.
5. For unevenly-sampled data (TESS has ~1-day gaps per orbit), tapify interpolates DPSS tapers to the actual timestamps.

## Fixes from original notebook

### 1. F-statistic normalization (was inconsistent)

**Original:** Cell 65 used `np.trapz(fs)` (missing frequency axis), and the main loop (cell 67) didn't normalize at all.

**Fix:** Area-normalize consistently with `np.trapz(final_fstat, final_freq)`, same as power.

### 2. ACF computation order (was distorted)

**Original:** ACF computed from interpolated, area-normalized power:
```
power → normPS(interp to 16K grid) → area normalize → ifft → ACF
```
The interpolation and renormalization change the spectral shape, so the resulting ACF doesn't reflect the true temporal autocorrelation.

**Fix:** Compute ACF from the original multi-taper power spectrum, then resample:
```
power → ifft → ACF(0)=1 normalize → resample to 2,500-point grid
```
This preserves the Wiener-Khinchin relationship between the original power spectrum and autocorrelation. The ACF is resampled to match the output grid length (2,500 bins) using unitless linspace interpolation — the conv encoder treats it as a 1D feature vector, so physical lag units are not needed.

## Normalized vs raw flux

**No effect.** `tapify.periodogram(center_data=True)` subtracts the mean internally. The H5 flux is z-scored (mean≈0, std=1), so centering is a no-op. Power scales as flux², but is area-normalized anyway. F-stat is a ratio (scale-invariant). ACF is normalized to ACF(0)=1 (scale-invariant).

## Assumptions

- **NW=4, K=7:** Resolution ~0.30 c/d for a 27-day sector. Smooths features narrower than that. Appropriate for broad spectral features; may blur sharp rotation peaks in short sectors. NW=2, K=3 would give better resolution (~0.15 c/d) but noisier estimates.
- **Max frequency 50 c/d:** Covers all age-relevant signals: rotation (0.05–2 c/d), granulation/flicker (0.1–10 c/d). Everything above ~50 c/d is photon noise and instrumental artifacts for typical TESS targets. The original notebook used 320 c/d (TESS Nyquist ≈ 360), but 95%+ of bins above 50 c/d are pure noise that the conv encoder would have to learn to ignore.
- **Output: 2,500 frequency bins** (50 c/d / 0.02 resolution). Reduced from the original 16,000 bins — 6× less storage (1.8 GB vs 11.5 GB total), faster I/O during training, with no loss of astrophysically useful information.
- **FFT method with irregular sampling:** tapify handles this via interpolated DPSS tapers. Assumes gaps aren't too severe (TESS orbits are fine).
- **Zero-padding to next power of 2:** Standard practice, increases frequency resolution.

## Files

| File | Purpose |
|------|---------|
| `scripts/compute_power_spectra.py` | Computation script — reads H5, writes power/f_stat/acf/freq back |
| `bin/data/compute_power_spectra.sh` | Shell wrapper with hardcoded parameters |

## Usage

```bash
# Run for both H5 files
./bin/data/compute_power_spectra.sh

# Debug with small subset
python scripts/compute_power_spectra.py \
    --h5_path final_pretrain/timeseries_pretrain.h5 \
    --max_samples 10
```
