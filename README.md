# LC-Gen: Light Curve Generation and Reconstruction

Multi-modal deep learning framework for stellar light curve analysis using autoencoders.

## Overview

LC-Gen is a research project focused on learning compressed latent representations of stellar light curves using various deep learning architectures. The project combines multiple data modalities (power spectra, autocorrelation functions, F-statistics, time series, and tabular data) to reconstruct and predict light curves.

## Features

- **Multiple Model Architectures**:
  - UNet-based CNN autoencoders for power spectra/ACF
  - MLP autoencoders for compact representations
  - Transformer models for time-series light curves

- **Multitaper Spectral Analysis** (NEW):
  - Compute PSD, ACF, and F-statistics from irregular time series
  - Integrated tapify for multitaper methods (NW, K parameters)
  - Proper handling of irregular sampling via mtNUFFT

- **Flexible Training**:
  - Masked autoencoding for robust representations
  - Chi-squared loss for uncertainty-aware training
  - Configurable via YAML files or Python API

- **Comprehensive Evaluation**:
  - Multiple metrics (MSE, MAE, chi-squared, R²)
  - Visualization utilities for reconstructions
  - Latent space analysis tools

## Installation

```bash
# Clone the repository
git clone https://github.com/pvl19/lc-gen.git
cd lc-gen

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Using Python API

```python
from lcgen.models import PowerSpectrumUNetAutoencoder, UNetConfig
from lcgen.data import normalize_ps
import torch

# Create model
config = UNetConfig(
    input_length=1024,
    encoder_dims=[64, 128, 256, 512],
    num_layers=4,
    activation='gelu'
)
model = PowerSpectrumUNetAutoencoder(config)

# Forward pass
power_spectrum = torch.randn(32, 1024)  # Batch of power spectra
output = model(power_spectrum)
reconstructed = output['reconstructed']
latent = output['encoded']
```

### Computing Spectral Features

```python
from lcgen.data import compute_multitaper_psd, psd_to_acf
import numpy as np

# Load light curve (time, flux, flux_err)
time = np.array([...])  # Irregular sampling
flux = np.array([...])
flux_err = np.array([...])

# Compute multitaper PSD and F-statistics
result = compute_multitaper_psd(
    time, flux, flux_err,
    NW=4.0,  # Time-bandwidth parameter
    K=7,      # Number of tapers
    return_fstat=True
)

# Extract features
psd = result['psd']
frequency = result['frequency']
fstat = result['fstat']

# Derive ACF from PSD
lags, acf = psd_to_acf(frequency, psd)
```

### Using CLI Scripts

```bash
# Compute spectra from light curves
python scripts/compute_spectra.py --input data/star_dataset.pt --NW 4.0 --fstat

# Train a model
python scripts/train.py --config configs/unet_ps.yaml

# Evaluate trained model
python scripts/evaluate.py --checkpoint experiments/results/best_model.pt --visualize
```

### Using Notebooks

Explore the `experiments/notebooks/` directory for interactive examples:
- `unet_exploration.ipynb` - UNet training and analysis
- `mlp_exploration.ipynb` - MLP autoencoder experiments
- `transformer_exploration.ipynb` - Transformer models for light curves

## Project Structure

```
lc-gen/
├── src/lcgen/              # Main package
│   ├── models/             # Model architectures
│   ├── data/               # Data processing utilities
│   ├── training/           # Training infrastructure
│   ├── evaluation/         # Metrics and visualization
│   └── utils/              # Configuration and logging
├── configs/                # YAML configuration files
├── scripts/                # CLI tools for training/evaluation
├── experiments/            # Notebooks and results
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── data/                   # Data directory (gitignored)
```

## Data Format

### Power Spectra / ACF
- Length: 1024 frequency bins
- Format: Normalized using arcsinh transformation
- File type: `.pt` (PyTorch tensors) or `.pickle`

### Light Curves
- Flux values with uncertainties
- Irregular time sampling
- Typical length: 720 timesteps (padded)

See [data/README.md](data/README.md) for detailed data format specifications.

## Long-term Vision

The project aims to develop a unified multi-modal architecture that:
1. Processes power spectra, ACF, and F-statistics via CNN branches
2. Handles irregular time series via transformers
3. Incorporates auxiliary tabular features via MLPs
4. Learns a shared latent representation
5. Reconstructs/predicts light curves with uncertainties
6. Supports forecasting and gap-filling

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architectural vision.

## Development

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for:
- How to add new models
- How to run tests
- How to benchmark performance
- Contribution guidelines

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lc_gen2025,
  title={LC-Gen: Multi-Modal Light Curve Reconstruction},
  author={LC-Gen Team},
  year={2025},
  url={https://github.com/pvl19/lc-gen}
}
```

## License

[Add license information]

## Contact

[Add contact information or link to issues]
