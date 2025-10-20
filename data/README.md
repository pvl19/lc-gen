# Data Directory

This directory contains training data for light curve reconstruction models.

## Mock Data (for development and testing)

### Mock Light Curves
- `mock_lightcurves/mock_lightcurves.pkl` - Generated synthetic light curves
  - 50,000 samples (default)
  - 4 LC types: sinusoid, multiperiodic, DRW, QPO
  - 4 sampling strategies: regular, random_gaps, variable_cadence, astronomical
  - Variable lengths: 200-1000 points

### Preprocessed Multi-Modal Features
- `mock_lightcurves/preprocessed.h5` - Multi-modal features (HDF5)
  - PSD, ACF, F-statistics (50000, 1024)
  - Tabular features (50000, 9)
  - Metadata for reconstruction

### Time Series Data
- `mock_lightcurves/timeseries.h5` - Padded time series for Transformer
  - Flux and time arrays (50000, 512)
  - Normalized for hierarchical Transformer

## Real Data (optional)

### Power Spectra and ACF Data
- `pk_star_lc_cf_norm.pickle` - Contains power spectra for stellar light curves
  - Format: Dictionary with keys being star IDs
  - Each entry contains: `power_1024_norm` (1024-length power spectrum)

### Light Curve Data
- `star_dataset.pt` - Full dataset of stellar light curves
- `star_dataset_subset.pt` - Subset for testing/development

## Data Format

### Power Spectrum Format
- Length: 1024 frequency bins
- Type: numpy float32 arrays
- Preprocessing: Normalized using `normalize_ps()` with arcsinh transformation

### ACF Format
- Length: 1024 time lags
- Type: numpy float32 arrays
- Preprocessing: Normalized using `normalize_acf()` with arcsinh transformation

### Light Curve Format (FluxDataset)
Each light curve contains:
- `flux`: Flux values (potentially with gaps/masking)
- `flux_err`: Uncertainty estimates for each flux measurement
- `time`: Timestamps (in days, irregular sampling)
- Typical sequence length: 720 timesteps (padded if necessary)

## Data Location

The data files are **not stored in the repository** (see `.gitignore`).

**Local Development**: Place data files in this directory
**Training Scripts**: Configure data paths in YAML config files under `configs/`

## Generating Mock Data

To generate mock data for development and testing:

### Step 1: Generate Light Curves
```bash
python scripts/generate_mock_data.py \
    --n_samples 50000 \
    --output_dir data/mock_lightcurves \
    --seed 42
```

### Step 2: Preprocess to Multi-Modal Features
```bash
# For MLP/UNet (spectral features)
python scripts/preprocess_mock_data.py \
    --input data/mock_lightcurves/mock_lightcurves.pkl \
    --output data/mock_lightcurves/preprocessed.h5 \
    --batch_size 100 \
    --verify

# For Transformer (time series)
python scripts/convert_timeseries_to_h5.py \
    --input data/mock_lightcurves/mock_lightcurves.pkl \
    --output data/mock_lightcurves/timeseries.h5 \
    --max_length 512
```

### Multi-Modal Preprocessing Pipeline

Uses the complete preprocessing pipeline with multitaper spectral analysis:

```python
from lcgen.data import preprocess_lightcurve_multimodal

result = preprocess_lightcurve_multimodal(
    time, flux, flux_err,
    NW=4.0,              # Time-bandwidth parameter
    K=7,                 # Number of tapers
    n_bins=1024          # Output resolution
)

# Access channels
acf = result['acf']           # (1024,)
psd = result['psd']           # (1024,)
fstat = result['fstat']       # (1024,)
timeseries = result['timeseries']  # (N, 2)
tabular = result['tabular']   # (9,)
```

See [../scripts/PREPROCESSING.md](../scripts/PREPROCESSING.md) for complete documentation.
