# Data Directory

This directory should contain the training data for light curve reconstruction models.

## Expected Data Files

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

## Generating Data

If you need to regenerate or preprocess data:

1. Place raw light curve data in this directory
2. Use preprocessing utilities from `src/data/preprocessing.py`
3. Save processed data in PyTorch `.pt` or pickle format

Example:
```python
from lcgen.data import normalize_ps
import pickle
import numpy as np

# Load raw data
with open('data/raw_power_spectra.pickle', 'rb') as f:
    raw_ps = pickle.load(f)

# Normalize
normalized_ps = np.array([normalize_ps(ps, scale_factor=50) for ps in raw_ps])

# Save
import torch
torch.save(torch.from_numpy(normalized_ps).float(), 'data/normalized_ps.pt')
```
