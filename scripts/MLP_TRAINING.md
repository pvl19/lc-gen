# MLP Autoencoder Training Guide

This guide covers training and evaluating MLP-based autoencoders on multi-modal light curve features.

## Overview

The MLP autoencoder uses a symmetric encoder-decoder architecture with:
- **Input**: 1024-dimensional feature vectors (PSD, ACF, or F-stat)
- **Architecture**: [1024 → 512 → 256 → 128 → 64] → [64 → 128 → 256 → 512 → 1024]
- **Training**: Self-supervised with block masking (70% ratio, 50-bin blocks)
- **Normalization**: Layer normalization + dropout (0.1)
- **Activation**: ReLU (configurable)

## Workflow

### 1. Generate Mock Data

```bash
python scripts/generate_mock_data.py \
    --n_samples 50000 \
    --output_dir data/mock_lightcurves \
    --seed 42
```

**Output**: `data/mock_lightcurves/mock_lightcurves.pkl`

**Properties**:
- 50,000 light curves
- 4 LC types: sinusoid (30%), multiperiodic (25%), DRW (25%), QPO (20%)
- 4 sampling: regular (40%), random_gaps (20%), variable_cadence (20%), astronomical (20%)
- Variable lengths: 200-1000 points
- Irregular sampling

### 2. Preprocess to Multi-Modal Features

```bash
python scripts/preprocess_mock_data.py \
    --input data/mock_lightcurves/mock_lightcurves.pkl \
    --output data/mock_lightcurves/preprocessed.h5 \
    --batch_size 100 \
    --NW 4.0 \
    --K 7 \
    --verify
```

**Output**: `data/mock_lightcurves/preprocessed.h5` (~660 MB compressed)

**HDF5 Structure**:
```
preprocessed.h5
├── acf           (50000, 1024)  - Autocorrelation function
├── psd           (50000, 1024)  - Power spectral density
├── fstat         (50000, 1024)  - F-statistics
├── pvalues       (50000, 1024)  - P-values for F-test
├── tabular       (50000, 9)     - Summary features
├── frequency     (1024,)        - Frequency grid
├── timeseries    (50000, max_len, 2) - Padded light curves
├── time          (50000, max_len) - Padded time arrays
├── mask          (50000, max_len) - Valid data mask
├── lc_type       (50000,)       - Light curve types
├── sampling_strategy (50000,)   - Sampling strategies
└── metadata/     (group)
    ├── median_flux (50000,)
    ├── iqr_flux    (50000,)
    ├── psd_scale_factor (50000,)
    ├── acf_scale_factor (50000,)
    ├── fstat_scale_factor (50000,)
    ├── psd_mean    (50000,)
    ├── psd_std     (50000,)
    ├── acf_mean    (50000,)
    ├── acf_std     (50000,)
    ├── fstat_mean  (50000,)
    └── fstat_std   (50000,)
```

**Normalization**:
- **PSD**: arcsinh(data / 1.0) → z-score
- **ACF**: arcsinh(data / 1.0) → z-score
- **F-stat**: arcsinh(data / 1.0) → z-score
- **Timeseries**: (flux - median) / IQR

### 3. Train Models

#### Option A: Train all three models sequentially

```bash
scripts\train_all_mlp.bat
```

#### Option B: Train individual models

**PSD Model:**
```bash
python scripts/train_mlp_autoencoder.py \
    --input data/mock_lightcurves/preprocessed.h5 \
    --channel psd \
    --output_dir models/mlp \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --mask_ratio 0.7 \
    --block_size 50
```

**ACF Model:**
```bash
python scripts/train_mlp_autoencoder.py \
    --input data/mock_lightcurves/preprocessed.h5 \
    --channel acf \
    --output_dir models/mlp \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --mask_ratio 0.7 \
    --block_size 50
```

**F-stat Model:**
```bash
python scripts/train_mlp_autoencoder.py \
    --input data/mock_lightcurves/preprocessed.h5 \
    --channel fstat \
    --output_dir models/mlp \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --mask_ratio 0.7 \
    --block_size 50
```

**Outputs** (per channel):
- `models/mlp/mlp_{channel}_best.pt` - Best model checkpoint
- `models/mlp/mlp_{channel}_final.pt` - Final model checkpoint
- `models/mlp/mlp_{channel}_training_curve.png` - Training curves
- `models/mlp/mlp_{channel}_recon_epoch{N}.png` - Reconstruction samples

**Training Time**: ~5-10 minutes per model on GPU, ~20-30 minutes on CPU (for 50K samples, 50 epochs)

### 4. Evaluate Models

```bash
# Evaluate PSD model
python scripts/evaluate_mlp.py \
    --input data/mock_lightcurves/preprocessed.h5 \
    --checkpoint models/mlp/mlp_psd_best.pt \
    --channel psd \
    --output_dir results/mlp

# Evaluate ACF model
python scripts/evaluate_mlp.py \
    --input data/mock_lightcurves/preprocessed.h5 \
    --checkpoint models/mlp/mlp_acf_best.pt \
    --channel acf \
    --output_dir results/mlp

# Evaluate F-stat model
python scripts/evaluate_mlp.py \
    --input data/mock_lightcurves/preprocessed.h5 \
    --checkpoint models/mlp/mlp_fstat_best.pt \
    --channel fstat \
    --output_dir results/mlp
```

**Outputs** (per channel):
- `results/mlp/mlp_{channel}_metrics.csv` - Quantitative metrics
- `results/mlp/mlp_{channel}_reconstruction_grid.png` - 16 sample reconstructions
- `results/mlp/mlp_{channel}_latent_space.png` - t-SNE visualization
- `results/mlp/mlp_{channel}_metrics_breakdown.png` - MSE/MAE by category

**Evaluation Time**: ~2-5 minutes per model (for 50K samples)

## Architecture Details

### Model Configuration

```python
MLPConfig(
    input_dim=1024,
    encoder_hidden_dims=[512, 256, 128],
    latent_dim=64,
    dropout=0.1,
    activation='relu'
)
```

**Parameter Count**: ~1.5M parameters

### Encoder
```
Input (1024)
  ↓ Linear + LayerNorm + ReLU + Dropout
Hidden1 (512)
  ↓ Linear + LayerNorm + ReLU + Dropout
Hidden2 (256)
  ↓ Linear + LayerNorm + ReLU + Dropout
Hidden3 (128)
  ↓ Linear (no activation)
Latent (64)
```

### Decoder
```
Latent (64)
  ↓ Linear + LayerNorm + ReLU + Dropout
Hidden1 (128)
  ↓ Linear + LayerNorm + ReLU + Dropout
Hidden2 (256)
  ↓ Linear + LayerNorm + ReLU + Dropout
Hidden3 (512)
  ↓ Linear (no activation)
Output (1024)
```

## Training Strategy

### Masking Framework

The model learns to reconstruct masked regions of spectral features:

1. **Block Masking**: Randomly select contiguous blocks to mask
   - `mask_ratio = 0.7`: Mask 70% of the 1024 bins
   - `block_size = 50`: Each masked block spans 50 bins
   - Number of blocks: ⌈1024 / 50⌉ × 0.7 ≈ 14 blocks

2. **Masking Operation**: Set masked regions to 0

3. **Loss Function**: MSE on masked regions only
   ```python
   loss = MSE(x_recon[mask], x_target[mask])
   ```

### Optimization

- **Optimizer**: AdamW with weight decay (1e-5)
- **Learning Rate**: 1e-3 with ReduceLROnPlateau scheduler
  - Factor: 0.5
  - Patience: 5 epochs
- **Batch Size**: 128
- **Train/Val Split**: 90% / 10%
- **Epochs**: 50

### Data Augmentation

Currently none - the masking itself provides regularization.

Future options:
- Random time shifts (circular permutation)
- Gaussian noise injection
- Mixup between samples

## Evaluation Metrics

### Quantitative

1. **Overall Performance**:
   - MSE (mean squared error)
   - MAE (mean absolute error)

2. **By Light Curve Type**:
   - Sinusoid performance
   - Multiperiodic performance
   - DRW performance
   - QPO performance

3. **By Sampling Strategy**:
   - Regular sampling
   - Random gaps
   - Variable cadence
   - Astronomical sampling

**Key Question**: Does irregular sampling degrade reconstruction quality?

### Qualitative

1. **Reconstruction Quality**:
   - Visual comparison of input vs. reconstructed spectra
   - Preservation of peaks and features
   - Smoothness of reconstruction

2. **Latent Space Structure**:
   - t-SNE visualization colored by LC type
   - t-SNE visualization colored by sampling strategy
   - Clustering and separation of different classes

3. **Feature Preservation**:
   - Are characteristic frequencies preserved?
   - Are QPO signatures maintained?
   - Are DRW red noise features captured?

## Expected Results

Based on preliminary analysis:

### PSD Autoencoder
- **Best for**: Sinusoids and multiperiodic signals (strong periodic features)
- **Expected MSE**: ~0.01-0.05 for periodic signals, ~0.1-0.2 for stochastic
- **Latent space**: Should cluster by period and amplitude

### ACF Autoencoder
- **Best for**: DRW signals (characteristic timescales in ACF)
- **Expected MSE**: ~0.05-0.15
- **Latent space**: Should cluster by correlation timescale

### F-stat Autoencoder
- **Best for**: Detecting periodic signals (high F-stat at true frequencies)
- **Expected MSE**: ~0.5-2.0 (F-stat has broader dynamic range)
- **Latent space**: Should separate periodic from non-periodic

### Sampling Effects
- **Regular sampling**: Best performance (baseline)
- **Random gaps**: Slight degradation (~10-20% higher MSE)
- **Variable cadence**: Moderate degradation (~20-30% higher MSE)
- **Astronomical**: Similar to variable cadence

## Troubleshooting

### High Training Loss

1. **Check data normalization**: Verify no NaNs or extreme values
   ```bash
   python scripts/preprocess_mock_data.py --verify
   ```

2. **Reduce learning rate**: Try 1e-4 instead of 1e-3

3. **Check masking**: Ensure 70% masking isn't too aggressive

### Poor Reconstruction Quality

1. **Increase latent dimension**: Try 128 instead of 64

2. **Add hidden layers**: Try [512, 384, 256, 128]

3. **Reduce dropout**: Try 0.05 instead of 0.1

### Overfitting

1. **Increase dropout**: Try 0.2

2. **Add weight decay**: Increase to 1e-4

3. **Reduce model capacity**: Smaller hidden dimensions

### NaN Loss

1. **Check for inf/nan in data**: Run verification
2. **Reduce learning rate**: Start with 1e-4
3. **Add gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

## Next Steps

After completing MLP training:

1. **CNN Autoencoder**: Use convolutional architecture for spectral features
2. **Transformer**: Self-attention for long-range dependencies
3. **Multi-Modal**: Joint training on PSD + ACF + F-stat
4. **Conditional**: Generate spectra conditioned on LC parameters
5. **Variational**: VAE for latent space sampling

## References

- PREPROCESSING.md - Complete preprocessing documentation
- CLAUDE.md - Development log and design decisions
- src/lcgen/models/mlp.py - Model implementation
- src/lcgen/data/ - Data processing pipeline
