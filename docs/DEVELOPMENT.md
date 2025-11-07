# Development Guide

This guide explains how to develop with the LC-Gen codebase, including adding new models, running tests, and contributing to the project.

## Table of Contents

1. [Setup](#setup)
2. [Project Structure](#project-structure)
3. [Adding New Models](#adding-new-models)
4. [Adding New Datasets](#adding-new-datasets)
5. [Training Models](#training-models)
6. [Testing](#testing)
7. [Benchmarking](#benchmarking)
8. [Code Style](#code-style)
9. [Contributing](#contributing)

## Setup

### Installation

```bash
# Clone the repository
git clone https://github.com/pvl19/lc-gen.git
cd lc-gen

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Data Setup

Place your data files in the `data/` directory:

```bash
data/
├── pk_star_lc_cf_norm.pickle       # Power spectra
├── star_dataset.pt                  # Light curves
└── star_dataset_subset.pt           # Subset for testing
```

See [data/README.md](../data/README.md) for data format specifications.

## Project Structure

```
lc-gen/
├── src/lcgen/                  # Main package
│   ├── models/                 # Model architectures
│   │   ├── base.py            # Base classes
│   │   ├── unet.py            # UNet autoencoder
│   │   ├── mlp.py             # MLP autoencoder
│   │   └── transformer.py     # Hierarchical Transformer (U-Net style)
│   ├── data/                   # Data processing
│   │   ├── datasets.py        # PyTorch datasets
│   │   ├── preprocessing.py   # Normalization functions
│   │   └── masking.py         # Masking strategies
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py         # Trainer class
│   │   ├── losses.py          # Loss functions
│   │   └── callbacks.py       # Training callbacks
│   ├── evaluation/             # Evaluation tools
│   │   ├── metrics.py         # Metrics computation
│   │   └── visualize.py       # Plotting utilities
│   └── utils/                  # Utilities
│       ├── config.py          # Configuration management
│       └── logging.py         # Logging utilities
├── configs/                    # YAML configurations
├── scripts/                    # CLI tools
│   ├── train_transformer_autoencoder.py  # Hierarchical Transformer training
│   ├── train_mlp_autoencoder.py         # MLP training
│   ├── train_unet_autoencoder.py        # UNet training
│   ├── generate_mock_data.py            # Mock LC generation
│   ├── preprocess_mock_data.py          # Multi-modal preprocessing
│   └── convert_timeseries_to_h5.py      # Time-series conversion
├── experiments/                # Notebooks and results
│   ├── notebooks/             # Jupyter notebooks
│   └── results/               # Saved models and logs
├── tests/                      # Unit tests
└── docs/                       # Documentation
```

## Adding New Models

### Step 1: Create Model Configuration

Define a configuration dataclass that extends `ModelConfig`:

```python
# src/lcgen/models/my_model.py
from dataclasses import dataclass
from .base import ModelConfig

@dataclass
class MyModelConfig(ModelConfig):
    model_type: str = "my_model"
    model_name: str = "my_autoencoder"

    # Your custom parameters
    hidden_dim: int = 256
    num_layers: int = 4
    # ... other config params
```

### Step 2: Implement Model Class

Extend `BaseAutoencoder` and implement required methods:

```python
from .base import BaseAutoencoder
import torch.nn as nn

class MyAutoencoder(BaseAutoencoder):
    def __init__(self, config: MyModelConfig):
        super().__init__(config)

        # Define your architecture
        self.encoder_module = ...
        self.decoder_module = ...

    def encode(self, x):
        """Encode input to latent representation"""
        return self.encoder_module(x)

    def decode(self, z):
        """Decode latent to output"""
        return self.decoder_module(z)

    def forward(self, x):
        """Full forward pass"""
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)

        return {
            'reconstructed': reconstructed,
            'encoded': encoded
        }
```

### Step 3: Register Model

Add your model to `src/lcgen/models/__init__.py`:

```python
from .my_model import MyAutoencoder, MyModelConfig

__all__ = [
    # ... existing models
    "MyAutoencoder",
    "MyModelConfig",
]
```

### Step 4: Create Configuration File

Create a YAML config in `configs/`:

```yaml
# configs/my_model.yaml
model:
  model_type: my_model
  hidden_dim: 256
  num_layers: 4

data:
  data_path: ./data
  batch_size: 32
  val_split: 0.2

training:
  epochs: 100
  learning_rate: 0.001
  loss_fn: mse

experiment_name: my_model_experiment
version: v01
```

### Step 5: Test Your Model

Create a unit test in `tests/test_models.py`:

```python
def test_my_autoencoder():
    config = MyModelConfig(hidden_dim=128, num_layers=2)
    model = MyAutoencoder(config)

    # Test forward pass
    x = torch.randn(4, 1024)
    output = model(x)

    assert output['reconstructed'].shape == x.shape
    assert output['encoded'].shape[0] == x.shape[0]
```

## Adding New Datasets

### Step 1: Create Dataset Class

Extend `torch.utils.data.Dataset`:

```python
# src/lcgen/data/datasets.py
from torch.utils.data import Dataset
import torch

class MyCustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = self.load_data(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_data(self, path):
        # Your data loading logic
        pass
```

### Step 2: Add to __init__.py

```python
# src/lcgen/data/__init__.py
from .datasets import MyCustomDataset

__all__ = [
    # ... existing datasets
    "MyCustomDataset",
]
```

### Step 3: Create Data Loader Utility

```python
from torch.utils.data import DataLoader

def create_my_dataset_loader(config):
    dataset = MyCustomDataset(
        data_path=config.data.data_path,
        transform=...
    )

    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers
    )
```

## Training Models

### Using Python API

```python
from lcgen.models import PowerSpectrumUNetAutoencoder, UNetConfig
from lcgen.training import Trainer
from lcgen.utils import load_config

# Load configuration
config = load_config('configs/unet_ps.yaml')

# Create model
model_config = UNetConfig(**config.model)
model = PowerSpectrumUNetAutoencoder(model_config)

# Create trainer (when implemented)
trainer = Trainer(model, config)

# Train
trainer.train(train_loader, val_loader)
```

### Using CLI (when implemented)

```bash
# Train with config file
python scripts/train.py --config configs/unet_ps.yaml

# Override config parameters
python scripts/train.py --config configs/unet_ps.yaml \
    --epochs 200 \
    --learning_rate 0.0001

# Evaluate trained model
python scripts/evaluate.py \
    --model experiments/results/unet/ps/v01/best_model.pt \
    --data data/test_ps.pt
```

### Using Notebooks

See `experiments/notebooks/` for interactive examples:

```python
# In Jupyter notebook
from lcgen.models import PowerSpectrumUNetAutoencoder, UNetConfig
from lcgen.data import normalize_ps, PowerSpectrumDataset
import torch

# Load and prepare data
data = load_power_spectra()
normalized_data = [normalize_ps(ps) for ps in data]
dataset = PowerSpectrumDataset(normalized_data)

# Create model
config = UNetConfig(
    input_length=1024,
    encoder_dims=[64, 128, 256, 512],
    num_layers=4
)
model = PowerSpectrumUNetAutoencoder(config)

# Training loop (simplified)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(epochs):
    for batch in train_loader:
        output = model(batch)
        loss = criterion(output['reconstructed'], batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src/lcgen tests/

# Run specific test
pytest tests/test_models.py::test_unet_forward_pass
```

### Writing Tests

Create test files in `tests/` with naming convention `test_*.py`:

```python
# tests/test_my_feature.py
import pytest
import torch
from lcgen.models import MyAutoencoder, MyModelConfig

def test_my_autoencoder_forward():
    """Test forward pass produces correct output shape"""
    config = MyModelConfig(hidden_dim=128)
    model = MyAutoencoder(config)

    x = torch.randn(4, 1024)
    output = model(x)

    assert 'reconstructed' in output
    assert output['reconstructed'].shape == x.shape

def test_my_autoencoder_encode():
    """Test encoding produces expected latent dimension"""
    config = MyModelConfig(latent_dim=64)
    model = MyAutoencoder(config)

    x = torch.randn(4, 1024)
    encoded = model.encode(x)

    assert encoded.shape[1] == config.latent_dim

@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_my_autoencoder_batch_sizes(batch_size):
    """Test model works with different batch sizes"""
    config = MyModelConfig()
    model = MyAutoencoder(config)

    x = torch.randn(batch_size, 1024)
    output = model(x)

    assert output['reconstructed'].shape[0] == batch_size
```

### Test Organization

```
tests/
├── test_models.py          # Model architecture tests
├── test_data.py            # Data processing tests
├── test_training.py        # Training loop tests
├── test_losses.py          # Loss function tests
├── test_metrics.py         # Evaluation metrics tests
└── benchmarks/
    ├── benchmark_unet.py   # Performance benchmarks
    └── benchmark_mlp.py
```

## Benchmarking

### Creating Benchmarks

```python
# tests/benchmarks/benchmark_my_model.py
import time
import torch
from lcgen.models import MyAutoencoder, MyModelConfig

def benchmark_inference_speed():
    """Benchmark inference speed"""
    config = MyModelConfig()
    model = MyAutoencoder(config).eval()

    x = torch.randn(32, 1024)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            _ = model(x)
        times.append(time.time() - start)

    print(f"Average inference time: {np.mean(times)*1000:.2f}ms")
    print(f"Throughput: {32/np.mean(times):.0f} samples/sec")

def benchmark_memory_usage():
    """Benchmark memory usage"""
    config = MyModelConfig()
    model = MyAutoencoder(config)

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Memory footprint (approximate)
    param_memory = params * 4 / 1024**2  # Float32 in MB
    print(f"Parameter memory: {param_memory:.2f} MB")

if __name__ == "__main__":
    benchmark_inference_speed()
    benchmark_memory_usage()
```

### Running Benchmarks

```bash
# Run single benchmark
python tests/benchmarks/benchmark_my_model.py

# Run all benchmarks
python -m pytest tests/benchmarks/ -v
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Group into stdlib, third-party, local
- **Docstrings**: Google style
- **Type hints**: Use where helpful (not required everywhere)

### Example Function

```python
def compute_reconstruction_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric: str = 'mse'
) -> float:
    """
    Compute reconstruction error between predictions and targets.

    Args:
        predictions: Model predictions of shape (batch, ...)
        targets: Ground truth targets of shape (batch, ...)
        metric: Error metric to use ('mse', 'mae', 'rmse')

    Returns:
        Scalar error value

    Raises:
        ValueError: If metric is not recognized

    Example:
        >>> pred = torch.randn(4, 1024)
        >>> targ = torch.randn(4, 1024)
        >>> error = compute_reconstruction_error(pred, targ, 'mse')
    """
    if metric == 'mse':
        return F.mse_loss(predictions, targets).item()
    elif metric == 'mae':
        return F.l1_loss(predictions, targets).item()
    elif metric == 'rmse':
        return torch.sqrt(F.mse_loss(predictions, targets)).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")
```

### Formatting Tools

```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy (optional)
mypy src/
```

## Contributing

### Contribution Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-new-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run tests**: `pytest`
6. **Commit changes**: `git commit -m "Add new feature"`
7. **Push to branch**: `git push origin feature/my-new-feature`
8. **Open a Pull Request**

### Pull Request Guidelines

- **Clear description**: Explain what your PR does and why
- **Tests**: Include tests for new features
- **Documentation**: Update docs if needed
- **Code style**: Follow the style guide
- **Small PRs**: Keep changes focused and manageable

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Example:
```
feat: add multi-head attention to transformer

Implement multi-head attention mechanism with configurable
number of heads. This improves model capacity for capturing
different aspects of temporal dependencies.

Closes #42
```

## Development Tips

### Debugging

```python
# Enable anomaly detection for NaN/Inf
torch.autograd.set_detect_anomaly(True)

# Set deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Reduce data for faster iteration
small_dataset = torch.utils.data.Subset(dataset, range(100))
```

### Profiling

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    model(input_data)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Monitoring GPU Usage

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

## Common Issues

### Out of Memory

- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing
- Use mixed precision training (FP16)

### Slow Training

- Use DataLoader with `num_workers > 0`
- Enable `pin_memory=True` for GPU
- Use compiled models (PyTorch 2.0+)
- Profile to find bottlenecks

### NaN Losses

- Check learning rate (may be too high)
- Check data normalization
- Add gradient clipping
- Check for division by zero in custom losses

## Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/
- **Weights & Biases**: https://docs.wandb.ai/ (for experiment tracking)
- **Ray Tune**: https://docs.ray.io/en/latest/tune/ (for hyperparameter tuning)

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions or share ideas
- **Documentation**: Check docs/ folder
- **Code Comments**: Read inline documentation

---

Happy developing! 🚀
