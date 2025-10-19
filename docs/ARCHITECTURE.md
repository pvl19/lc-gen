# Architecture Vision

This document describes the long-term architectural vision for the LC-Gen project, including the multi-modal deep learning framework for light curve reconstruction and prediction.

## Current State (v0.1.0)

The project currently implements **separate autoencoder architectures** trained independently on different data modalities:

### Implemented Models

1. **UNet Autoencoder** (`PowerSpectrumUNetAutoencoder`)
   - **Input**: Power spectra or ACF (1024 frequency bins)
   - **Architecture**: CNN with skip connections
   - **Latent**: Spatial bottleneck features (channels × compressed spatial dimension)
   - **Use case**: Learning compressed representations of frequency-domain information

2. **MLP Autoencoder** (`MLPAutoencoder`)
   - **Input**: Power spectra or ACF (1024 bins)
   - **Architecture**: Fully-connected encoder-decoder
   - **Latent**: Compact vector (typically 64-128 dimensions)
   - **Use case**: Highly compressed representations for analysis

3. **Transformer Models** (`TimeSeriesTransformer`, `MaskedTimeSeriesTransformer`)
   - **Input**: Light curves (flux + uncertainty, irregular sampling)
   - **Architecture**: Transformer encoder with time-aware positional encoding
   - **Latent**: Sequence of embeddings (seq_len × d_model)
   - **Use case**: Modeling temporal dependencies and masked reconstruction

### Current Training Paradigm

- **Autoencoding objective**: Reconstruct input from latent representation
- **Masked training**: Random contiguous blocks masked during training
- **Loss functions**:
  - MSE for power spectra/ACF
  - Chi-squared for light curves (weighted by uncertainties)
- **Training**: Each model trained independently on its modality

## Future Vision: Multi-Modal Architecture

The ultimate goal is a **unified multi-modal encoder-decoder** that processes multiple data sources simultaneously to reconstruct complete light curves.

### Target Architecture

```
                    ┌─────────────────────────────────┐
                    │   Multi-Modal Encoder Network   │
                    └─────────────────────────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬───────────────┐
        │               │           │           │               │
   ┌────▼────┐    ┌────▼────┐ ┌───▼────┐ ┌────▼─────┐   ┌────▼────┐
   │   ACF   │    │   PSD   │ │ F-stat │ │TimeSeries│   │ Tabular │
   │ Branch  │    │ Branch  │ │ Branch │ │  Branch  │   │ Branch  │
   │ (UNet)  │    │ (UNet)  │ │ (UNet) │ │(Transformer)│ │  (MLP)  │
   └────┬────┘    └────┬────┘ └───┬────┘ └────┬─────┘   └────┬────┘
        │               │           │           │               │
        └───────────────┴───────────┼───────────┴───────────────┘
                                    │
                             ┌──────▼──────┐
                             │   Fusion    │
                             │   Layer     │
                             │ (Attention  │
                             │   or Concat)│
                             └──────┬──────┘
                                    │
                          ┌─────────▼─────────┐
                          │  Shared Latent    │
                          │  Representation   │
                          │  (z_t : temporal) │
                          └─────────┬─────────┘
                                    │
                             ┌──────▼──────┐
                             │   Decoder   │
                             │  Network    │
                             │(Transformer)│
                             └──────┬──────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Reconstructed     │
                          │  Light Curve       │
                          │  (flux + σ_flux)   │
                          └────────────────────┘
```

### Multi-Modal Input Branches

#### 1. ACF Branch (Autocorrelation Function)
- **Architecture**: UNet CNN encoder
- **Input**: ACF (1024 time lags)
- **Output**: Latent vector `z_acf` (dim: 256-512)
- **Purpose**: Capture temporal correlation structure

#### 2. PSD Branch (Power Spectral Density)
- **Architecture**: UNet CNN encoder
- **Input**: Power spectrum (1024 frequency bins)
- **Output**: Latent vector `z_psd` (dim: 256-512)
- **Purpose**: Capture frequency-domain characteristics

#### 3. F-statistic Branch
- **Architecture**: UNet CNN encoder (similar to PSD)
- **Input**: F-statistic from multitaper analysis (via tapify)
- **Output**: Latent vector `z_fstat` (dim: 256-512)
- **Purpose**: Detect and characterize periodic signals

#### 4. Time Series Branch
- **Architecture**: Transformer encoder
- **Input**: Partially observed light curve (flux, time, uncertainty)
- **Output**: Sequence latent `z_ts` (seq_len × d_model)
- **Purpose**: Model temporal dynamics and handle irregular sampling

#### 5. Tabular Branch
- **Architecture**: MLP encoder
- **Input**: Auxiliary features (stellar parameters, survey metadata, etc.)
- **Output**: Latent vector `z_tab` (dim: 64-128)
- **Purpose**: Incorporate physical constraints and context

### Fusion Strategy

**Option A: Concatenation Fusion**
```
z_combined = concat([z_acf, z_psd, z_fstat, flatten(z_ts), z_tab])
```
- Simple, interpretable
- Fixed-size representation
- Requires all inputs (or handling missing modalities)

**Option B: Cross-Attention Fusion**
```
z_fused = CrossAttention(
    query=z_ts,           # Time series as query
    keys=[z_acf, z_psd, z_fstat, z_tab],  # Other modalities as keys
    values=[z_acf, z_psd, z_fstat, z_tab]
)
```
- More flexible
- Learns importance of each modality
- Handles missing modalities naturally

**Option C: Hierarchical Fusion**
```
z_freq = Fusion([z_psd, z_fstat])      # Frequency domain
z_time = Fusion([z_acf, z_ts])          # Time domain
z_combined = Fusion([z_freq, z_time, z_tab])
```
- Groups related modalities
- May learn better representations

**Recommendation**: Start with Option A for simplicity, transition to Option B/C as needed.

### Decoder Architecture

**Transformer Decoder** with temporal modeling:
- Input: Fused latent `z_combined` + target timestamps
- Architecture: Transformer decoder with causal masking (for forecasting)
- Outputs:
  - **Flux predictions**: `f(t)` at arbitrary timestamps
  - **Uncertainties**: `σ_f(t)` (aleatoric uncertainty)
  - Optional: **Forecasts**: `f(t + Δt)` for future times

### Latent Space Design

The shared latent representation should:

1. **Temporal Evolution**: Model `z_t` as a function of time
   - Use state-space models or temporal embeddings
   - Enable forecasting by extrapolating `z_{t+Δt}`

2. **Interpretability**: Components should correspond to:
   - Periodicity information (from PSD/F-stat)
   - Amplitude/noise characteristics (from ACF)
   - Trends and transients (from time series)
   - Physical constraints (from tabular)

3. **Compactness**: Target dimension ~256-512 for full latent
   - Smaller than concatenating all branches independently
   - Learned compression via fusion layer

## Training Strategy

### Phase 1: Individual Branch Pre-training (CURRENT)
- Train each branch independently on its modality
- Use autoencoding objectives
- Build strong feature extractors

### Phase 2: Multi-Modal Joint Training (FUTURE)
- Freeze or fine-tune pre-trained branches
- Train fusion layer and decoder end-to-end
- Objective: Reconstruct full light curves from all inputs

**Loss Function**:
```
L_total = L_recon + λ_unc * L_uncertainty + λ_reg * L_regularization

where:
  L_recon = χ² loss on flux reconstruction
  L_uncertainty = Uncertainty calibration loss
  L_regularization = KL divergence or other latent regularization
```

### Phase 3: Advanced Capabilities
1. **Forecasting**: Extend decoder to predict future light curve values
2. **Gap-filling**: Mask portions of time series, predict missing values
3. **Uncertainty Quantification**: Learn predictive distributions
4. **Transfer Learning**: Fine-tune on specific stellar types

## Data Requirements

### Input Data Modalities

| Modality | Format | Length | Preprocessing | Source |
|----------|--------|--------|---------------|--------|
| Power Spectrum (PSD) | 1D array | 1024 | arcsinh normalization | FFT of light curve |
| Autocorrelation (ACF) | 1D array | 1024 | arcsinh normalization | IFFT of PSD |
| F-statistic | 1D array | 1024 | TBD | Multitaper analysis (tapify) |
| Light Curve | (time, flux, σ) | 720 (padded) | Standardization | TESS/Kepler observations |
| Tabular | Feature vector | Variable | Standardization | Stellar catalogs (Gaia, etc.) |

### Missing Modality Handling

The architecture should gracefully handle missing inputs:

1. **Zero-padding**: Replace missing modality with zeros
2. **Learned embeddings**: Use trainable "missing" token
3. **Attention masking**: Mask out missing modalities in fusion
4. **Modality dropout**: During training, randomly drop modalities (for robustness)

## Implementation Roadmap

### v0.2.0 - Unified Data Pipeline
- [ ] Implement `MultiModalDataset` with all modalities
- [ ] Create data loaders that handle missing modalities
- [ ] Add F-statistic computation pipeline (tapify integration)
- [ ] Standardize tabular feature extraction

### v0.3.0 - Multi-Modal Encoder
- [ ] Implement fusion layer (concatenation-based)
- [ ] Create `MultiModalEncoder` class combining all branches
- [ ] Enable loading pre-trained branch weights
- [ ] Test forward pass with all/subset of modalities

### v0.4.0 - Full Architecture
- [ ] Implement Transformer decoder for light curve reconstruction
- [ ] Add uncertainty prediction heads
- [ ] Implement combined loss functions
- [ ] End-to-end training pipeline

### v0.5.0 - Advanced Features
- [ ] Forecasting capabilities
- [ ] Improved uncertainty quantification
- [ ] Cross-attention fusion (replacing concatenation)
- [ ] Benchmarking and ablation studies

### v1.0.0 - Production-Ready
- [ ] Optimized inference
- [ ] Model serving infrastructure
- [ ] Documentation and tutorials
- [ ] Published results and paper

## Research Questions

1. **Optimal Fusion**: Which fusion strategy works best for light curve reconstruction?

2. **Modality Importance**: Which input modalities contribute most to reconstruction quality?

3. **Latent Interpretability**: Can we interpret latent dimensions in terms of physical parameters?

4. **Transfer Learning**: Does pre-training on power spectra improve light curve forecasting?

5. **Uncertainty Calibration**: How well-calibrated are the uncertainty estimates?

6. **Computational Efficiency**: Can we achieve real-time inference for large catalogs?

## Technical Considerations

### Memory and Compute
- Multi-modal models are memory-intensive
- Consider gradient checkpointing for deep models
- Use mixed-precision training (FP16) where possible
- Implement efficient attention mechanisms (e.g., Flash Attention)

### Scalability
- Batch processing across modalities
- Distributed training for large datasets
- Model parallelism for very large models

### Reproducibility
- Fixed random seeds
- Deterministic operations where possible
- Version control for data, code, and model checkpoints
- Configuration management via YAML

### Interpretability
- Attention visualization (which modalities are important?)
- Latent space analysis (PCA, t-SNE)
- Ablation studies (remove modalities, measure impact)
- Feature attribution methods

## References

- **UNet Architecture**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **Transformers**: Vaswani et al., "Attention Is All You Need" (2017)
- **Multi-Modal Learning**: Ngiam et al., "Multimodal Deep Learning" (2011)
- **Uncertainty in Deep Learning**: Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning?" (2017)
- **Time Series Transformers**: Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (2021)

## Contact and Contributions

For questions about the architecture or to propose modifications, please open an issue on GitHub or contact the project maintainers.
