# Legacy Code Archive

This directory contains the original code structure before the repository reorganization (v0.1.0).

**Archived on**: 2025-10-19
**Reason**: Repository reorganized into modular package structure

## Contents

### Old Model Implementations

- **convolution/** - Original UNet implementations for power spectra
  - Contains CNN-based autoencoder code
  - Predecessor to `src/lcgen/models/unet.py`

- **mlp/** - Original MLP autoencoder implementations
  - Contains fully-connected autoencoder code
  - Predecessor to `src/lcgen/models/mlp.py`

- **transformer/** - Original transformer implementations for light curves
  - Contains time-series transformer code
  - Predecessor to `src/lcgen/models/transformer.py`

- **Modules/** - Original module definitions
  - Shared utilities and helper functions
  - Functionality migrated to `src/lcgen/`

### Old Scripts and Configurations

- **data_prep.py** - Original preprocessing script
  - Migrated to `src/lcgen/data/preprocessing.py`

- **mlp_models.py** - Old MLP model definitions
  - Migrated to `src/lcgen/models/mlp.py`

- **mlp_training.py** - Old training script for MLPs
  - Replaced by `scripts/train.py` with unified Trainer

- **power_spectrum_config.yaml** - Old configuration file
  - Replaced by YAML configs in `configs/`

- **ps_training.ipynb** - Old training notebook
  - Reference for updating notebooks in `experiments/notebooks/`

## Migration Map

| Old Location | New Location | Status |
|-------------|--------------|---------|
| `convolution/` | `src/lcgen/models/unet.py` | ✓ Migrated |
| `mlp/` | `src/lcgen/models/mlp.py` | ✓ Migrated |
| `transformer/` | `src/lcgen/models/transformer.py` | ✓ Migrated |
| `Modules/` | `src/lcgen/` (various modules) | ✓ Migrated |
| `data_prep.py` | `src/lcgen/data/preprocessing.py` | ✓ Enhanced |
| `mlp_models.py` | `src/lcgen/models/mlp.py` | ✓ Migrated |
| `mlp_training.py` | `scripts/train.py` | ✓ Unified |
| `power_spectrum_config.yaml` | `configs/*.yaml` | ✓ Expanded |
| `ps_training.ipynb` | `experiments/notebooks/` | ⧗ To be updated |

## Why This Code Was Archived

The original code structure had several limitations:

1. **No unified interface** - Each model type had different APIs
2. **Scattered utilities** - Preprocessing and training code duplicated
3. **Hard to test** - No modular structure for unit testing
4. **Difficult to extend** - Adding new models required duplicating infrastructure
5. **No pip installation** - Code had to be imported via sys.path hacks

The new structure addresses all these issues with:
- Unified `BaseAutoencoder` interface for all models
- Modular package organization (`src/lcgen/`)
- Comprehensive training infrastructure with callbacks
- YAML configuration system
- Pip-installable package
- CLI tools for common operations

## Using Legacy Code

If you need to reference the old implementations:

1. **For model architecture details**: Check the old implementations against new ones
2. **For training notebooks**: Use as reference when updating `experiments/notebooks/`
3. **For hyperparameters**: Check old configs against new YAML files

**Note**: The legacy code is preserved for reference only and is not maintained. All new development should use the reorganized structure in `src/lcgen/`.

## Should I Use This Code?

**No** - Use the new package structure instead:

```python
# OLD (don't use)
# sys.path.insert(0, 'legacy/mlp')
# from mlp_models import MLPAutoencoder

# NEW (use this)
from lcgen.models import MLPAutoencoder, MLPConfig
```

See the main [README.md](../README.md) and [docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md) for current usage.

---

**Last Updated**: 2025-10-19
**Status**: Archived (read-only reference)
