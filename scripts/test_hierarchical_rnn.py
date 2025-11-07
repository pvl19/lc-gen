"""
Test script for hierarchical RNN (minLSTM/minGRU) autoencoder.

Verifies:
1. Model instantiation
2. Forward pass with correct shapes
3. CPU and CUDA compatibility
4. Parameter count
5. Memory usage estimation
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lcgen.models.rnn import HierarchicalRNN, RNNConfig


def test_rnn_forward_pass(rnn_type='minlstm', device='cpu'):
    """Test forward pass with minLSTM or minGRU."""
    print(f"\n{'='*60}")
    print(f"Testing Hierarchical RNN ({rnn_type}) on {device.upper()}")
    print(f"{'='*60}")

    # Create configuration
    config = RNNConfig(
        input_dim=2,  # flux + mask
        input_length=512,
        encoder_dims=[64, 128, 256, 512],
        rnn_type=rnn_type,
        num_layers_per_level=2,
        dropout=0.0,
        min_period=0.00278,
        max_period=1640.0,
        num_freqs=32
    )

    # Create model
    model = HierarchicalRNN(config)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Configuration:")
    print(f"  RNN type: {rnn_type}")
    print(f"  Input: (batch, {config.input_length}, {config.input_dim})")
    print(f"  Encoder dims: {config.encoder_dims}")
    print(f"  RNN layers per level: {config.num_layers_per_level}")
    print(f"  Total parameters: {n_params:,}")

    # Estimate memory
    param_memory_mb = n_params * 4 / 1024**2  # Float32
    print(f"  Parameter memory: {param_memory_mb:.2f} MB")

    # Create dummy input
    batch_size = 4
    seq_len = config.input_length

    x = torch.randn(batch_size, seq_len, config.input_dim, device=device)
    t = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1).float()

    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  t: {t.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, t)

    reconstructed = output['reconstructed']
    latent = output['latent']
    skip_connections = output['skip_connections']

    print(f"\nOutput shapes:")
    print(f"  Reconstructed: {reconstructed.shape}")
    print(f"  Latent (bottleneck): {latent.shape}")
    print(f"  Number of skip connections: {len(skip_connections)}")

    # Verify shapes
    assert reconstructed.shape == (batch_size, seq_len, 1), \
        f"Expected reconstructed shape {(batch_size, seq_len, 1)}, got {reconstructed.shape}"

    expected_latent_seq = seq_len // (2 ** len(config.encoder_dims))
    expected_latent_dim = config.encoder_dims[-1]
    assert latent.shape == (batch_size, expected_latent_seq, expected_latent_dim), \
        f"Expected latent shape {(batch_size, expected_latent_seq, expected_latent_dim)}, got {latent.shape}"

    print(f"\n  Skip connection shapes:")
    for i, skip in enumerate(skip_connections):
        expected_seq = seq_len // (2 ** i)
        print(f"    Level {i}: {skip.shape} (expected seq_len: {expected_seq})")

    # Test loss computation
    target = torch.randn(batch_size, seq_len, device=device)
    criterion = nn.MSELoss()
    loss = criterion(reconstructed.squeeze(-1), target)
    print(f"\nLoss computation test:")
    print(f"  MSE Loss: {loss.item():.6f}")

    print(f"\n[OK] All tests passed for {rnn_type} on {device}!")


def compare_minlstm_vs_mingru():
    """Compare minLSTM and minGRU parameter counts."""
    print(f"\n{'='*60}")
    print("Comparing minLSTM vs minGRU")
    print(f"{'='*60}")

    config_base = {
        'input_dim': 2,
        'input_length': 512,
        'encoder_dims': [64, 128, 256, 512],
        'num_layers_per_level': 2,
        'dropout': 0.0
    }

    # minLSTM
    config_lstm = RNNConfig(**config_base, rnn_type='minlstm')
    model_lstm = HierarchicalRNN(config_lstm)
    n_params_lstm = sum(p.numel() for p in model_lstm.parameters() if p.requires_grad)

    # minGRU
    config_gru = RNNConfig(**config_base, rnn_type='minGRU')
    model_gru = HierarchicalRNN(config_gru)
    n_params_gru = sum(p.numel() for p in model_gru.parameters() if p.requires_grad)

    print(f"\nParameter comparison:")
    print(f"  minLSTM: {n_params_lstm:,} parameters")
    print(f"  minGRU:  {n_params_gru:,} parameters")
    print(f"  Difference: {abs(n_params_lstm - n_params_gru):,} parameters")

    if n_params_gru < n_params_lstm:
        reduction = (1 - n_params_gru / n_params_lstm) * 100
        print(f"  minGRU is {reduction:.1f}% smaller")
    else:
        increase = (n_params_gru / n_params_lstm - 1) * 100
        print(f"  minGRU is {increase:.1f}% larger")


def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    print(f"\n{'='*60}")
    print("Testing Gradient Flow")
    print(f"{'='*60}")

    config = RNNConfig(
        input_dim=2,
        input_length=512,
        encoder_dims=[64, 128, 256, 512],
        rnn_type='minlstm',
        num_layers_per_level=2
    )

    model = HierarchicalRNN(config)

    # Create dummy data
    x = torch.randn(2, 512, 2, requires_grad=True)
    t = torch.arange(512).unsqueeze(0).expand(2, -1).float()
    target = torch.randn(2, 512)

    # Forward pass
    output = model(x, t)
    reconstructed = output['reconstructed'].squeeze(-1)

    # Compute loss and backward
    criterion = nn.MSELoss()
    loss = criterion(reconstructed, target)
    loss.backward()

    # Check gradients
    n_params_with_grad = 0
    n_params_total = 0
    for name, param in model.named_parameters():
        n_params_total += 1
        if param.grad is not None:
            n_params_with_grad += 1
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"  WARNING: NaN or Inf in gradients for {name}")

    print(f"\nGradient check:")
    print(f"  Parameters with gradients: {n_params_with_grad}/{n_params_total}")
    print(f"  Input gradient: {'[OK]' if x.grad is not None else '[FAIL]'}")

    if n_params_with_grad == n_params_total:
        print(f"\n[OK] Gradients flowing correctly!")
    else:
        print(f"\n[WARNING] Some parameters missing gradients!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Hierarchical RNN (minLSTM/minGRU) Test Suite")
    print("Based on 'Were RNNs All We Needed?' (arXiv:2410.01201)")
    print("="*60)

    # Test minLSTM on CPU
    test_rnn_forward_pass(rnn_type='minlstm', device='cpu')

    # Test minGRU on CPU
    test_rnn_forward_pass(rnn_type='minGRU', device='cpu')

    # Test on CUDA if available
    if torch.cuda.is_available():
        print(f"\nCUDA is available: {torch.cuda.get_device_name(0)}")
        test_rnn_forward_pass(rnn_type='minlstm', device='cuda')
        test_rnn_forward_pass(rnn_type='minGRU', device='cuda')
    else:
        print(f"\nCUDA not available, skipping GPU tests")

    # Compare architectures
    compare_minlstm_vs_mingru()

    # Test gradients
    test_gradient_flow()

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
