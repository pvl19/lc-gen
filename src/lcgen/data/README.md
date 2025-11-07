# Data Module

This module provides data processing utilities for astronomical light curve analysis.

## Modules

### preprocessing.py

Complete preprocessing pipeline for transforming raw light curves into multi-modal features.

**Main Functions:**
- `preprocess_lightcurve_multimodal()` - Full pipeline: raw LC → 5 channels
- `batch_preprocess_multimodal()` - Batch processing for multiple light curves
- `preprocess_lightcurve_for_spectral()` - Clean and standardize for spectral analysis
- `normalize_spectral_features()` - Normalize PSD/ACF/F-stat channels

**Legacy Functions:**
- `normalize_ps()` / `denormalize_ps()` - Power spectrum normalization
- `normalize_acf()` - ACF normalization

See [PREPROCESSING.md](../../../PREPROCESSING.md) for complete documentation.

### spectral.py

Multitaper spectral analysis utilities using the `tapify` package.

**Main Functions:**
- `compute_multitaper_psd()` - Multitaper PSD with optional F-statistics
- `psd_to_acf()` - ACF from PSD via Wiener-Khinchin theorem
- `compute_frequency_grid()` - Log/linear frequency grids with dynamic bounds
- `fstatistic_to_pvalue()` - Convert F-statistics to p-values
- `extract_acf_timescale()` - Characteristic timescale from ACF
- `resample_acf_to_bins()` - Interpolate ACF to fixed grid

**Key Features:**
- Thomson F-test for harmonic detection
- Non-uniform FFT for irregular sampling
- Adaptive weighting of tapers

### datasets.py

PyTorch dataset classes for loading astronomical data.

**Classes:**
- `FluxDataset` - Light curves with flux and uncertainties
- `PowerSpectrumDataset` - Power spectra or ACF data
- `MultiModalDataset` - Combined multi-modal features

### masking.py

Masking strategies for self-supervised learning.

**Functions:**
- `block_predefined_mask()` - Contiguous block masking for 1D data
- `block_predefined_mask_lc()` - Light curve masking (flux + uncertainties)

## Quick Start

### Multi-Modal Preprocessing

```python
from lcgen.data import preprocess_lightcurve_multimodal

# Raw light curve data
time = np.array([...])     # MJD or similar
flux = np.array([...])     # Flux measurements
flux_err = np.array([...]) # Uncertainties

# Preprocess to 5 channels
result = preprocess_lightcurve_multimodal(
    time, flux, flux_err,
    NW=4.0, K=7, n_bins=1024
)

# Access channels
acf = result['acf']        # (1024,)
psd = result['psd']        # (1024,)
fstat = result['fstat']    # (1024,)
ts = result['timeseries']  # (N, 2)
tab = result['tabular']    # (9,)
```

### Spectral Analysis

```python
from lcgen.data import compute_multitaper_psd

# Compute PSD and F-statistics
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

### Batch Processing

```python
from lcgen.data import batch_preprocess_multimodal

batch_result = batch_preprocess_multimodal(
    times=[time1, time2, time3],
    fluxes=[flux1, flux2, flux3],
    flux_errs=[err1, err2, err3],
    NW=4.0, K=7
)

# Batch outputs: (N_samples, n_bins)
acf_batch = batch_result['acf']     # (3, 1024)
psd_batch = batch_result['psd']     # (3, 1024)
fstat_batch = batch_result['fstat'] # (3, 1024)
```

## Output Channels

### 1. Autocorrelation Function (ACF)

- **Shape**: (1024,)
- **Content**: Temporal correlation structure
- **Normalization**: arcsinh transformation
- **Mode**: Correlation (ACF(0)=1) or covariance

### 2. Power Spectral Density (PSD)

- **Shape**: (1024,)
- **Content**: Frequency-domain power distribution
- **Normalization**: arcsinh transformation
- **Method**: Multitaper with NUFFT

### 3. F-statistic

- **Shape**: (1024,)
- **Content**: Thomson F-test for periodic signals
- **Normalization**: z-score (standardized)
- **Distribution**: F(2, 2K-2)

### 4. Time Series

- **Shape**: (N, 2) where N is number of observations
- **Channel 0**: Standardized flux (median/IQR)
- **Channel 1**: Scaled uncertainties (flux_err / IQR)

### 5. Tabular Features

- **Shape**: (9,)
- **Features**:
  1. median_flux - Position parameter
  2. iqr_flux - Scale parameter
  3. psd_integral - Total power
  4. psd_peak - Peak power
  5. acf_variance - Variance
  6. acf_timescale - Characteristic timescale
  7. lc_duration - Observation duration
  8. n_observations - Number of points
  9. median_cadence - Median sampling rate

## Dependencies

- **tapify** - Multitaper spectral analysis (GitHub installation required)
- **nfft** - Non-uniform FFT (required by tapify)
- **statsmodels** - Statistical tests (required by tapify)
- **scipy** - Scientific computing
- **numpy** - Array operations
- **torch** - PyTorch datasets

## References

See [PREPROCESSING.md](../../../PREPROCESSING.md) for detailed methodology and references.
