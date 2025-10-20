# LC-Gen: Light Curve Generation and Reconstruction

Multi-modal deep learning framework for stellar light curve analysis using autoencoders.

## Overview

LC-Gen is a research project focused on learning compressed latent representations of stellar light curves using various deep learning architectures. The project combines multiple data modalities (power spectra, autocorrelation functions, F-statistics, and time series) to reconstruct and predict light curves.

## Features

- **Multiple Model Architectures**:
  - **Hierarchical U-Net Transformer** - Multi-scale transformer with skip connections for time-series
  - **UNet CNN Autoencoder** - Convolutional architecture for power spectra/ACF
  - **MLP Autoencoder** - Fully-connected networks for compact representations

- **Optimized Training Pipeline**:
  - Fast dynamic block masking with power-of-2 sampling (3.2x speedup)
  - OneCycleLR scheduler for efficient training
  - Gradient clipping for stable transformer training
  - Unified output format across all models

- **Multitaper Spectral Analysis**:
  - Compute PSD, ACF, and F-statistics from irregular time series
  - Integrated tapify for multitaper methods (NW, K parameters)
  - Proper handling of irregular sampling via mtNUFFT

- **Flexible Training**:
  - Masked autoencoding for robust representations
  - Configurable via command-line arguments or YAML files
  - Separate losses for masked/unmasked regions

- **Comprehensive Evaluation**:
  - Multiple metrics (MSE on masked/unmasked regions)
  - Visualization utilities for reconstructions
  - Training curve plotting

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

### 1. Generate Mock Data

```bash
# Generate 50K light curves
python scripts/generate_mock_data.py --n_samples 50000 --output_dir data/mock_lightcurves

# Preprocess for MLP/UNet (spectral features)
python scripts/preprocess_mock_data.py \
  --input data/mock_lightcurves/mock_lightcurves.pkl \
  --output data/mock_lightcurves/preprocessed.h5 \
  --batch_size 100

# Convert for Transformer (time-series)
python scripts/convert_timeseries_to_h5.py \
  --input data/mock_lightcurves/mock_lightcurves.pkl \
  --output data/mock_lightcurves/timeseries.h5 \
  --max_length 512
```

### 2. Train Models

```bash
# Train MLP autoencoder on PSD
python scripts/train_mlp_autoencoder.py \
  --channel psd \
  --epochs 50 \
  --batch_size 128 \
  --lr 1e-3

# Train UNet autoencoder on ACF
python scripts/train_unet_autoencoder.py \
  --channel acf \
  --epochs 50 \
  --batch_size 128

# Train Hierarchical Transformer on time-series
python scripts/train_transformer_autoencoder.py \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4
```

All scripts use sensible defaults and will automatically use the mock data if available.

### 3. Using Python API

```python
from lcgen.models import TimeSeriesTransformer, TransformerConfig
import torch

# Create hierarchical transformer
config = TransformerConfig(
    input_dim=2,  # flux + mask
    input_length=512,
    encoder_dims=[64, 128, 256, 512],  # Multi-scale hierarchy
    nhead=4,
    num_layers=4,
    num_transformer_blocks=2  # Blocks per resolution
)
model = TimeSeriesTransformer(config)

# Forward pass
x = torch.randn(32, 512, 2)  # [batch, seq_len, channels]
t = torch.randn(32, 512)      # timestamps
output = model(x, t)
reconstructed = output['reconstructed']  # (32, 512, 1)
encoded = output['encoded']              # (32, 32, 512) bottleneck
```

## Model Architectures

### Hierarchical Transformer

**Key Features:**
- **Multi-scale processing** - Attention at 512, 256, 128, 64, 32 sequence lengths
- **Skip connections** - Like UNet, preserves fine-scale information
- **Pre-norm architecture** - Better training stability
- **Time-aware positional encoding** - Log-spaced frequencies for astrophysical timescales
- **~34% fewer attention operations** compared to flat transformers

**Architecture:**
```
Input (512 length, 2 channels)
  ↓ Embedding (64 dim)
  ↓ Positional Encoding
  ↓ Level 0: Transform @ 64 dim → Skip → Downsample (256 length, 128 dim)
  ↓ Level 1: Transform @ 128 dim → Skip → Downsample (128 length, 256 dim)
  ↓ Level 2: Transform @ 256 dim → Skip → Downsample (64 length, 512 dim)
  ↓ Level 3: Transform @ 512 dim → Skip → Downsample (32 length, 512 dim)
  ↓ Bottleneck: Transform @ 512 dim (32 length)
  ↓ Upsample + Skip fusion (64 length, 512 dim)
  ↓ Decode Level 3: Transform @ 512 dim
  ↓ Upsample + Skip fusion (128 length, 256 dim)
  ↓ ... (continue back to original resolution)
Output (512 length, 1 channel)
```

**Parameters:** ~26.8M

### UNet Autoencoder

**Key Features:**
- Progressive downsampling with skip connections
- GELU activation, LayerNorm
- Explicit mask channel
- Optimized for spectral data (PSD/ACF/Fstat)

**Configuration:**
```python
UNetConfig(
    input_length=1024,
    in_channels=2,  # data + mask
    encoder_dims=[64, 128, 256, 512],
    num_layers=4,
    activation='gelu'
)
```

### MLP Autoencoder

**Key Features:**
- Symmetric encoder-decoder
- Compact latent bottleneck
- Fast training and inference
- LayerNorm for stability

**Configuration:**
```python
MLPConfig(
    input_dim=2048,  # data + mask concatenated
    encoder_hidden_dims=[512, 256, 128],
    latent_dim=64,
    activation='gelu'
)
```

## Performance Optimizations

### Fast Dynamic Block Masking

We use an optimized masking strategy:
- **Power-of-2 block sizes** for log-style distribution
- **No validation overhead** - samples directly without pre-computing valid combinations
- **3.2x speedup** compared to validated masking (97-104ms → 33ms per batch)
- Applied per-sample in collate function for flexibility

### Training Speed

On 50K light curves (RTX 5080):
- **MLP**: ~14 sec/epoch
- **UNet**: ~20-30 sec/epoch (estimated)
- **Transformer**: ~60-90 sec/epoch (hierarchical architecture saves ~33% vs flat)

### OneCycleLR Scheduler

All training scripts use OneCycleLR for faster convergence:
```python
OneCycleLR(
    optimizer,
    max_lr=lr,
    total_steps=epochs * steps_per_epoch,
    pct_start=0.1,       # 10% warmup
    div_factor=1e2,      # Initial LR = max_lr / 100
    final_div_factor=1e2 # Final LR = max_lr / 10000
)
```

## Project Structure

```
lc-gen/
├── src/lcgen/              # Main package
│   ├── models/             # Model architectures
│   │   ├── transformer.py # Hierarchical Transformer (NEW)
│   │   ├── unet.py        # UNet autoencoder
│   │   ├── mlp.py         # MLP autoencoder
│   │   └── base.py        # Base classes
│   ├── data/               # Data processing
│   │   ├── preprocessing.py # Multitaper spectral analysis
│   │   ├── masking.py      # Fast dynamic block masking (OPTIMIZED)
│   │   └── datasets.py     # PyTorch datasets
│   └── utils/              # Utilities
├── scripts/                # Training and data preparation
│   ├── train_transformer_autoencoder.py  # Transformer training (NEW)
│   ├── train_mlp_autoencoder.py         # MLP training
│   ├── train_unet_autoencoder.py        # UNet training
│   ├── generate_mock_data.py            # Mock LC generation
│   ├── preprocess_mock_data.py          # Spectral preprocessing
│   └── convert_timeseries_to_h5.py      # Time-series conversion
├── configs/                # YAML configuration files
│   ├── transformer_lc.yaml
│   ├── mlp_ps.yaml
│   └── unet_ps.yaml
├── models/                 # Saved model checkpoints
└── data/                   # Data directory (gitignored)
    └── mock_lightcurves/   # Generated mock data
```

## Data Format

### Spectral Features (PSD/ACF/Fstat)
- **Format**: HDF5 with keys `['psd', 'acf', 'fstat']`
- **Shape**: (N, 1024) for N light curves
- **Normalization**: Arcsinh transformation
- **Used by**: MLP and UNet autoencoders

### Time Series
- **Format**: HDF5 with keys `['flux', 'time']`
- **Shape**: (N, 512) for N light curves
- **Normalization**: Median-centered, IQR-scaled
- **Used by**: Hierarchical Transformer

## Training Output

All training scripts produce consistent output:

```
Using device: cuda:0

Loading data from data/mock_lightcurves/preprocessed.h5...

Dataset statistics:
  Shape: (50000, 1024)
  Range: [-3.562, 29.530]
  Mean: -0.000
  Std: 1.000

Dataset split:
  Train: 45000 samples
  Val: 5000 samples

Creating Hierarchical Transformer autoencoder...
Model parameters: 26,812,289

Starting training for 50 epochs...
Dynamic masking: block size [1, 50], mask ratio [10%, 70%]

Epoch 1/50
  Train - Loss: 0.123456 | Masked: 0.234567 | Unmasked: 0.012345
  Val   - Loss: 0.234567 | Masked: 0.345678 | Unmasked: 0.023456
  Saved best model to models/transformer/transformer_best.pt
...
```

## Long-term Vision

The project aims to develop a unified multi-modal architecture that:
1. Processes power spectra, ACF, and F-statistics via CNN branches ✓
2. Handles irregular time series via hierarchical transformers ✓
3. Learns a shared latent representation (In progress)
4. Reconstructs/predicts light curves with uncertainties (In progress)
5. Supports forecasting and gap-filling (Future)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lc_gen2025,
  title={LC-Gen: Multi-Modal Light Curve Reconstruction with Hierarchical Transformers},
  author={LC-Gen Team},
  year={2025},
  url={https://github.com/pvl19/lc-gen}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

[Add contact information or link to issues]
