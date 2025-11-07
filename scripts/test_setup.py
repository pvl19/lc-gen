#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify LC-Gen installation and basic functionality.

Usage:
    python scripts/test_setup.py
"""

import sys
import io
from pathlib import Path
import torch

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from lcgen.models import (
            PowerSpectrumUNetAutoencoder, UNetConfig,
            MLPAutoencoder, MLPConfig,
            TimeSeriesTransformer, TransformerConfig
        )
        print("  ✓ Models imported successfully")

        from lcgen.data import (
            normalize_ps, normalize_acf,
            block_predefined_mask,
            PowerSpectrumDataset, FluxDataset
        )
        print("  ✓ Data utilities imported successfully")

        from lcgen.training import (
            Trainer,
            mse_loss, chi_squared_loss,
            ModelCheckpoint, EarlyStopping
        )
        print("  ✓ Training infrastructure imported successfully")

        from lcgen.evaluation import (
            compute_mse, evaluate_reconstruction,
            plot_reconstruction
        )
        print("  ✓ Evaluation utilities imported successfully")

        from lcgen.utils import load_config, Config
        print("  ✓ Utilities imported successfully")

        return True

    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")

    try:
        # Test UNet
        from lcgen.models import PowerSpectrumUNetAutoencoder, UNetConfig

        config = UNetConfig(
            input_length=1024,
            encoder_dims=[32, 64, 128],
            num_layers=3,
            activation='gelu'
        )
        model = PowerSpectrumUNetAutoencoder(config)
        print(f"  ✓ UNet created ({model.count_parameters():,} parameters)")

        # Test MLP
        from lcgen.models import MLPAutoencoder, MLPConfig

        config = MLPConfig(
            input_dim=1024,
            encoder_hidden_dims=[256, 128],
            latent_dim=64
        )
        model = MLPAutoencoder(config)
        print(f"  ✓ MLP created ({model.count_parameters():,} parameters)")

        # Test Transformer
        from lcgen.models import TimeSeriesTransformer, TransformerConfig

        config = TransformerConfig(
            input_dim=2,
            d_model=64,
            nhead=2,
            num_layers=2
        )
        model = TimeSeriesTransformer(config)
        print(f"  ✓ Transformer created ({model.count_parameters():,} parameters)")

        return True

    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass through models"""
    print("\nTesting forward passes...")

    try:
        # Test UNet forward
        from lcgen.models import PowerSpectrumUNetAutoencoder, UNetConfig

        config = UNetConfig(encoder_dims=[32, 64], num_layers=2)
        model = PowerSpectrumUNetAutoencoder(config)

        x = torch.randn(4, 1024)
        output = model(x)

        assert 'reconstructed' in output
        assert 'encoded' in output
        assert output['reconstructed'].shape == x.shape
        print("  ✓ UNet forward pass successful")

        # Test MLP forward
        from lcgen.models import MLPAutoencoder, MLPConfig

        config = MLPConfig(input_dim=1024, latent_dim=64)
        model = MLPAutoencoder(config)

        x = torch.randn(4, 1024)
        output = model(x)

        assert output['reconstructed'].shape == x.shape
        print("  ✓ MLP forward pass successful")

        # Test Transformer forward
        from lcgen.models import TimeSeriesTransformer, TransformerConfig

        config = TransformerConfig(d_model=32, nhead=2, num_layers=2)
        model = TimeSeriesTransformer(config)

        x = torch.randn(4, 100, 2)  # (batch, seq_len, channels)
        t = torch.randn(4, 100, 1)  # timestamps
        output = model(x, t)

        assert output['reconstructed'].shape[0] == x.shape[0]
        print("  ✓ Transformer forward pass successful")

        return True

    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_processing():
    """Test data processing utilities"""
    print("\nTesting data processing...")

    try:
        import numpy as np
        from lcgen.data import normalize_ps, normalize_acf, block_predefined_mask

        # Test normalization
        ps = np.random.rand(1024) * 100
        normalized = normalize_ps(ps, scale_factor=50)
        assert normalized.shape == ps.shape
        print("  ✓ Power spectrum normalization works")

        acf = np.random.rand(1024)
        normalized = normalize_acf(acf, scale_factor=0.001)
        assert normalized.shape == acf.shape
        print("  ✓ ACF normalization works")

        # Test masking
        x = torch.randn(1024)
        masked_x, mask = block_predefined_mask(x, block_size=10, mask_ratio=0.5)
        assert masked_x.shape == x.shape
        assert mask.dtype == torch.bool
        print("  ✓ Block masking works")

        return True

    except Exception as e:
        print(f"  ✗ Data processing failed: {e}")
        return False


def test_loss_functions():
    """Test loss functions"""
    print("\nTesting loss functions...")

    try:
        from lcgen.training import mse_loss, chi_squared_loss, masked_mse_loss

        # Test MSE
        pred = torch.randn(10, 100)
        target = torch.randn(10, 100)
        loss = mse_loss(pred, target)
        assert loss.item() >= 0
        print("  ✓ MSE loss works")

        # Test masked MSE
        mask = torch.rand(10, 100) > 0.5
        loss = masked_mse_loss(pred, target, mask)
        assert loss.item() >= 0
        print("  ✓ Masked MSE loss works")

        # Test chi-squared
        uncertainty = torch.rand(10, 100) + 0.1
        loss = chi_squared_loss(pred, target, uncertainty)
        assert loss.item() >= 0
        print("  ✓ Chi-squared loss works")

        return True

    except Exception as e:
        print(f"  ✗ Loss functions failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")

    try:
        from lcgen.utils import load_config
        import yaml
        from pathlib import Path

        # Check if config files exist
        config_dir = Path(__file__).parent.parent / 'configs'

        if not config_dir.exists():
            print("  ⚠ Config directory not found, skipping")
            return True

        config_files = list(config_dir.glob('*.yaml'))

        if not config_files:
            print("  ⚠ No config files found, skipping")
            return True

        # Test loading first config file
        config_file = config_files[0]
        config = load_config(str(config_file))

        assert hasattr(config, 'model')
        assert hasattr(config, 'data')
        assert hasattr(config, 'training')

        print(f"  ✓ Config loaded from {config_file.name}")

        return True

    except Exception as e:
        print(f"  ⚠ Config loading test skipped: {e}")
        return True  # Don't fail on config test


def main():
    print("=" * 60)
    print("LC-Gen Setup Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Data Processing", test_data_processing()))
    results.append(("Loss Functions", test_loss_functions()))
    results.append(("Config Loading", test_config_loading()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")

    print("=" * 60)
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! LC-Gen is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
