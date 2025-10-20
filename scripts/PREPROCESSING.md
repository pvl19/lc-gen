# Multi-Modal Light Curve Preprocessing

This document describes the complete preprocessing pipeline that transforms raw irregularly-sampled light curves into multi-modal features for the LC-Gen encoder network.

## Overview

The preprocessing pipeline takes raw astronomical time series data and produces 5 distinct input channels:

1. **Autocorrelation Function (ACF)** - Temporal correlation structure
2. **Power Spectral Density (PSD)** - Frequency-domain power distribution
3. **F-statistic** - Statistical significance of periodic signals (Thomson F-test)
4. **Time Series** - Standardized flux and scaled uncertainties
5. **Tabular Features** - Auxiliary summary statistics

## Pipeline Workflow

### Complete 10-Step Process

```python
from lcgen.data import preprocess_lightcurve_multimodal

result = preprocess_lightcurve_multimodal(
    time, flux, flux_err,
    NW=4.0,              # Time-bandwidth parameter
    K=7,                 # Number of tapers
    n_bins=1024,         # Output resolution
    freq_spacing='log',  # Log-spaced frequency grid
    compute_pvalues=True # Compute F-test p-values
)
```

The function performs these steps:

1. **Clean and standardize** - Remove NaN, outliers; standardize using median/IQR
2. **Compute frequency grid** - Log or linear spaced, dynamic bounds
3. **Compute multitaper PSD** - Using tapify with NUFFT method
4. **Compute F-statistic** - Thomson F-test for harmonic detection
5. **Compute p-values** - F(2, 2K-2) distribution for significance testing
6. **Derive ACF from PSD** - Wiener-Khinchin theorem via inverse FFT
7. **Resample to 1024 bins** - Uniform grid via interpolation
8. **Extract auxiliary features** - PSD integral/peak, ACF variance/timescale
9. **Normalize spectral features** - arcsinh for PSD/ACF, z-score for F-stat
10. **Assemble output channels** - 5 channels ready for encoder

## Multitaper Spectral Analysis

### What is Multitaper?

The multitaper method uses multiple orthogonal tapers (DPSS - Discrete Prolate Spheroidal Sequences) to estimate the power spectral density. This approach:

- **Reduces spectral leakage** compared to single-window methods
- **Balances bias and variance** through averaging multiple independent estimates
- **Handles irregular sampling** via Non-Uniform FFT (NUFFT)
- **Provides statistical tests** for periodic signal detection

### Key Parameters

**NW (Time-Bandwidth Product)**
- Controls frequency resolution
- Typical values: 2.5, 3.0, 3.5, 4.0
- Higher values → better frequency resolution but more variance
- Resolution bandwidth: W = NW / T, where T is duration

**K (Number of Tapers)**
- Number of independent spectral estimates
- Typical: K = 2*NW - 1 or K = 2*NW
- More tapers → lower variance but slightly increased bias
- Must have K ≥ 2 for F-test

**Method**
- `'fft'` - Fast NUFFT (default, recommended for irregular sampling)
- `'dft'` - Direct DFT (slower but more accurate)
- `'ls'` - Lomb-Scargle (faster but no F-statistics)

### Thomson F-Test

The F-statistic tests the null hypothesis that the signal at each frequency is purely noise against the alternative that it contains a harmonic component.

**F-distribution**: F ~ F(2, 2K-2)
- dfn = 2 (numerator degrees of freedom)
- dfd = 2K-2 (denominator degrees of freedom)

**P-values**: Computed using the survival function of the F-distribution
- p < 0.01 → highly significant periodic signal
- p < 0.05 → significant periodic signal

**Interpretation**:
- F > 10-20 typically indicates strong periodicity
- P-values account for multiple testing via frequency resolution

## Robust Standardization

### Why Median/IQR?

We use **robust statistics** instead of mean/std to handle:

- Outliers from cosmic rays, instrumental artifacts
- Non-Gaussian flux distributions
- Heavy-tailed noise in astronomical data

**Standardization formula**:
```
flux_standardized = (flux - median) / IQR
```

where IQR = Q3 - Q1 (interquartile range)

**Advantages**:
- Resistant to outliers (up to 25% contamination)
- Preserves signal structure
- More stable for sparse/irregular sampling

## Autocorrelation Function

### ACF from PSD via Wiener-Khinchin

The Wiener-Khinchin theorem states that the autocorrelation function is the inverse Fourier transform of the power spectral density:

```
ACF(τ) = IFFT(PSD(f))
```

**Implementation details**:
1. Compute PSD on frequency grid
2. Apply inverse FFT to get ACF in time domain
3. Resample ACF to lag grid matching 1024 bins
4. Normalize to correlation (ACF(0) = 1) or keep as covariance

### ACF Modes

**Correlation (default)**:
- Normalized: ACF(0) = 1
- Represents correlation coefficient vs lag
- Variance saved separately in metadata

**Covariance**:
- Preserves amplitude: ACF(0) = variance
- Represents covariance vs lag

### Extracting Timescale

Three methods to extract characteristic timescale from ACF:

1. **E-folding** (default): Lag where ACF drops to 1/e ≈ 0.368
2. **Half-max**: Lag where ACF drops to 0.5
3. **Zero-crossing**: First lag where ACF crosses zero

## Normalization Strategy

All normalizations use **arcsinh transformation** followed by **z-score** for neural network compatibility.

### PSD: arcsinh + z-score

```python
psd_asinh = arcsinh(psd_raw / psd_scale_factor)
psd_normalized = (psd_asinh - mean) / std
```

**Default**: `psd_scale_factor = 1.0`

**Why arcsinh?**
- Logarithmic behavior for large values (dynamic range compression)
- Linear behavior near zero (preserves low-power features)
- Smooth and differentiable everywhere
- No issues with zero or negative values (unlike log)

**Tuning**: Lower scale_factor → more compression of large values

### ACF: arcsinh + z-score

```python
acf_asinh = arcsinh(acf_raw / acf_scale_factor)
acf_normalized = (acf_asinh - mean) / std
```

**Default**: `acf_scale_factor = 1.0`

**Why arcsinh?**
- Symmetric around zero (handles negative ACF values)
- Compresses extreme positive/negative correlations
- Preserves structure near zero

**Tuning**: Lower scale_factor → more compression of extreme values

### F-statistic: arcsinh + z-score (optional)

```python
# Option 1: Z-score only (default, backward compatible)
fstat_normalized = (fstat - mean) / (std + ε)

# Option 2: Arcsinh + z-score (recommended for training)
fstat_asinh = arcsinh(fstat / fstat_scale_factor)
fstat_normalized = (fstat_asinh - mean) / std
```

**Default**: `fstat_scale_factor = 1.0` (arcsinh + z-score) ✓ Recommended
**Legacy**: `fstat_scale_factor = None` (z-score only, backward compatible)

**Why arcsinh for F-stat?**
- F-statistics can have extreme values (>100) for strong periodic signals
- Arcsinh compresses these outliers (e.g., 100 → 5.3 after arcsinh)
- Better dynamic range for neural networks
- Typical output range: [-2, 4] vs [-2, 30] without arcsinh

**Impact of arcsinh normalization**:
```
Without arcsinh (None): F-stat range = [-1.21, 6.27]
With arcsinh (1.0):     F-stat range = [-1.64, 3.31]  ✓ Better for training
```

### Configuring Normalization

```python
result = preprocess_lightcurve_multimodal(
    time, flux, flux_err,
    psd_scale_factor=1.0,     # PSD compression
    acf_scale_factor=1.0,     # ACF compression
    fstat_scale_factor=1.0,   # Enable arcsinh for F-stat
)
```

## Frequency Grid

### Dynamic Bounds

**Minimum frequency** (longest period):
```
f_min = 1 / (t_max - t_min)  # One cycle over observation duration
```

**Maximum frequency** (Nyquist):
```
f_max = 0.5 / median(Δt)  # Nyquist frequency from median cadence
```

### Spacing Options

**Log spacing** (default):
```python
freq = np.logspace(log10(f_min), log10(f_max), n_bins)
```
- Better resolution at low frequencies (long periods)
- Preferred for astronomical variability (power-law PSDs common)
- Matches human perception of frequency

**Linear spacing**:
```python
freq = np.linspace(f_min, f_max, n_bins)
```
- Uniform frequency resolution
- Better for evenly-distributed signals

## Tabular Features (9 total)

### Light Curve Statistics (3)
1. **median_flux** - Central position (robust to outliers)
2. **iqr_flux** - Scale parameter (robust spread)
3. **lc_duration** - Total time span (days)

### PSD Features (2)
4. **psd_integral** - Total power (integral of PSD)
5. **psd_peak** - Maximum power (strongest frequency component)

### ACF Features (2)
6. **acf_variance** - Variance of standardized flux
7. **acf_timescale** - Characteristic correlation timescale (days)

### Sampling Properties (2)
8. **n_observations** - Number of data points
9. **median_cadence** - Median time between observations (days)

## Output Format

### Dictionary Structure

```python
result = {
    # 5 Input Channels
    'acf': (1024,),           # Normalized ACF
    'psd': (1024,),           # Normalized PSD
    'fstat': (1024,),         # Normalized F-statistic
    'timeseries': (N, 2),     # [flux_std, err_scaled]
    'time': (N,),             # Time array for positional encoding
    'tabular': (9,),          # Auxiliary features

    # Additional Outputs
    'pvalues': (1024,),       # F-test p-values
    'frequency': (1024,),     # Frequency grid

    # Metadata (for reconstruction)
    'metadata': {
        'median_flux': float,
        'iqr_flux': float,
        'psd_integral': float,
        'psd_peak': float,
        'acf_variance': float,
        'acf_timescale': float,
        'lc_duration': float,
        'n_observations': int,
        'median_cadence': float,
        'NW': float,
        'K': int,
        'freq_spacing': str,
        'acf_mode': str,
    }
}
```

## Batch Processing

For processing multiple light curves:

```python
from lcgen.data import batch_preprocess_multimodal

batch_result = batch_preprocess_multimodal(
    times=[time1, time2, ...],      # List of time arrays
    fluxes=[flux1, flux2, ...],     # List of flux arrays
    flux_errs=[err1, err2, ...],    # List of error arrays
    NW=4.0,
    K=7,
    verbose=True
)

# Output shapes:
# batch_result['acf']:   (N_samples, 1024)
# batch_result['psd']:   (N_samples, 1024)
# batch_result['fstat']: (N_samples, 1024)
# batch_result['tabular']: (N_samples, 9)
# batch_result['timeseries']: list of N_samples arrays (variable length)
```

## Usage Examples

### Basic Usage

```python
import numpy as np
from lcgen.data import preprocess_lightcurve_multimodal

# Load your light curve
time = np.array([...])       # MJD or similar
flux = np.array([...])       # Flux measurements
flux_err = np.array([...])   # Uncertainties

# Preprocess
result = preprocess_lightcurve_multimodal(time, flux, flux_err)

# Access channels
acf = result['acf']           # (1024,)
psd = result['psd']           # (1024,)
fstat = result['fstat']       # (1024,)
timeseries = result['timeseries']  # (N, 2)
tabular = result['tabular']   # (9,)

# Find significant periodic signals
sig_freqs = result['frequency'][result['pvalues'] < 0.01]
print(f"Significant frequencies: {sig_freqs}")
```

### Custom Parameters

```python
result = preprocess_lightcurve_multimodal(
    time, flux, flux_err,
    NW=3.5,                    # Narrower bandwidth
    K=6,                       # Fewer tapers
    n_bins=2048,               # Higher resolution
    freq_spacing='linear',     # Linear frequency grid
    freq_min=0.001,            # Custom frequency range
    freq_max=5.0,
    acf_mode='covariance',     # Preserve amplitude
    outlier_sigma=3.0,         # More aggressive outlier removal
)
```

### Spectral Analysis Only

For just computing PSD and F-statistics:

```python
from lcgen.data import compute_multitaper_psd

psd_result = compute_multitaper_psd(
    time, flux,
    NW=4.0,
    K=7,
    return_fstat=True
)

freq = psd_result['frequency']
psd = psd_result['psd']
fstat = psd_result['fstat']
```

## Computational Notes

### Performance

- Single light curve (~500 points): ~0.5-1 second
- Batch of 100 light curves: ~50-100 seconds
- Bottleneck: NUFFT computation in tapify

### Memory

- Memory usage scales with n_bins and K
- Typical: ~10 MB per light curve for full preprocessing
- Batch processing: n_samples × 10 MB + overhead

### Dependencies

- **tapify** - Multitaper PSD (installed from GitHub)
- **nfft** - Non-uniform FFT (required by tapify)
- **statsmodels** - Multiple testing corrections (required by tapify)
- **scipy** - F-distribution, interpolation
- **numpy** - Array operations

## References

1. Thomson, D. J. (1982). "Spectrum estimation and harmonic analysis." *Proceedings of the IEEE*, 70(9), 1055-1096.
2. Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for Physical Applications*. Cambridge University Press.
3. Patil, A. A., et al. (2022). "The Multitaper Spectrum Analysis Package in Python." *Seismological Research Letters*, 93(3), 1922-1929.

## See Also

- [src/lcgen/data/README.md](src/lcgen/data/README.md) - Data module documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [scripts/test_multimodal_preprocessing.py](scripts/test_multimodal_preprocessing.py) - Comprehensive test suite
