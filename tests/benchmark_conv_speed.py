"""Benchmark performance impact of convolutional channels."""
import torch
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lcgen.models.simple_min_gru import BiDirectionalMinGRU
import numpy as np


def benchmark_model(model, batch_size, seq_length, freq_length, use_conv, num_iterations=50, warmup=5):
    """Benchmark model forward pass speed."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # Create dummy data
    flux = torch.randn(batch_size, seq_length, device=device)
    flux_err = torch.randn(batch_size, seq_length, device=device)
    times = torch.randn(batch_size, seq_length, device=device)
    metadata = torch.randn(batch_size, 8, device=device)  # 8 metadata features
    x_in = torch.stack([flux, flux_err], dim=-1)
    t_in = times.unsqueeze(-1)

    # Create conv data if needed
    if use_conv:
        conv_data = {
            'power': torch.randn(batch_size, freq_length, device=device),
            'f_stat': torch.randn(batch_size, freq_length, device=device),
            'acf': torch.randn(batch_size, freq_length, device=device)
        }
    else:
        conv_data = None

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x_in, t_in, metadata=metadata, conv_data=conv_data)

    # Benchmark
    times_ms = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(x_in, t_in, metadata=metadata, conv_data=conv_data)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times_ms.append((end - start) * 1000)

    return np.array(times_ms)


if __name__ == '__main__':
    print("="*60)
    print("Performance Benchmark: Conv Channels Impact")
    print("="*60)

    # Configuration
    hidden_size = 64
    direction = 'bi'
    num_meta_features = 8
    batch_size = 16
    seq_length = 1024
    freq_length = 16000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Frequency length: {freq_length}")

    # Baseline model
    print("\n" + "-"*60)
    print("Benchmarking BASELINE model (no conv channels)...")
    print("-"*60)

    model_base = BiDirectionalMinGRU(
        hidden_size=hidden_size,
        direction=direction,
        use_flow=False,
        num_meta_features=num_meta_features,
        use_conv_channels=False
    )

    times_base = benchmark_model(model_base, batch_size, seq_length, freq_length, use_conv=False)
    mean_base = np.mean(times_base)
    std_base = np.std(times_base)

    print(f"Baseline: {mean_base:.2f} ± {std_base:.2f} ms/batch")

    # Model with LIGHTWEIGHT conv channels (recommended)
    print("\n" + "-"*60)
    print("Benchmarking model WITH LIGHTWEIGHT conv channels...")
    print("-"*60)

    conv_config_light = {
        'encoder_type': 'lightweight',
        'hidden_channels': 16,
        'num_layers': 3,
        'activation': 'gelu'
    }

    model_conv_light = BiDirectionalMinGRU(
        hidden_size=hidden_size,
        direction=direction,
        use_flow=False,
        num_meta_features=num_meta_features,
        use_conv_channels=True,
        conv_config=conv_config_light
    )

    times_conv_light = benchmark_model(model_conv_light, batch_size, seq_length, freq_length, use_conv=True)
    mean_conv_light = np.mean(times_conv_light)
    std_conv_light = np.std(times_conv_light)

    print(f"Lightweight conv: {mean_conv_light:.2f} ± {std_conv_light:.2f} ms/batch")

    # Model with UNET conv channels (for comparison)
    print("\n" + "-"*60)
    print("Benchmarking model WITH UNET conv channels (slower)...")
    print("-"*60)

    conv_config_unet = {
        'encoder_type': 'unet',
        'input_length': 16000,
        'encoder_dims': [4, 8, 16, 32],
        'num_layers': 4,
        'activation': 'gelu'
    }

    model_conv_unet = BiDirectionalMinGRU(
        hidden_size=hidden_size,
        direction=direction,
        use_flow=False,
        num_meta_features=num_meta_features,
        use_conv_channels=True,
        conv_config=conv_config_unet
    )

    times_conv_unet = benchmark_model(model_conv_unet, batch_size, seq_length, freq_length, use_conv=True)
    mean_conv_unet = np.mean(times_conv_unet)
    std_conv_unet = np.std(times_conv_unet)

    print(f"UNet conv: {mean_conv_unet:.2f} ± {std_conv_unet:.2f} ms/batch")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline model:          {mean_base:.2f} ms/batch")
    print(f"Lightweight conv:        {mean_conv_light:.2f} ms/batch  ({100*(mean_conv_light - mean_base)/mean_base:.1f}% overhead)")
    print(f"UNet conv:               {mean_conv_unet:.2f} ms/batch  ({100*(mean_conv_unet - mean_base)/mean_base:.1f}% overhead)")
    print(f"\nThroughput comparison:")
    print(f"  Baseline:      {1000/mean_base:.1f} batches/sec = {1000*batch_size/mean_base:.0f} samples/sec")
    print(f"  Lightweight:   {1000/mean_conv_light:.1f} batches/sec = {1000*batch_size/mean_conv_light:.0f} samples/sec")
    print(f"  UNet:          {1000/mean_conv_unet:.1f} batches/sec = {1000*batch_size/mean_conv_unet:.0f} samples/sec")

    # Breakdown of where time is spent
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    print("\nThe conv encoders run ONCE per sample (not per timestep), so:")
    print(f"  - Lightweight overhead: ~{(mean_conv_light - mean_base)/seq_length:.4f} ms/timestep")
    print(f"  - UNet overhead: ~{(mean_conv_unet - mean_base)/seq_length:.4f} ms/timestep")
    print(f"  - This is amortized across {seq_length} timesteps")
    print("\nThe RNN still dominates runtime since it processes sequentially.")
    print("\n⚡ RECOMMENDATION: Use 'lightweight' encoder (default) for best performance!")
    print("="*60)
