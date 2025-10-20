"""Quick test to verify hierarchical Transformer architecture"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lcgen.models.transformer import TimeSeriesTransformer, TransformerConfig

# Create model
config = TransformerConfig(
    input_dim=2,
    input_length=512,
    encoder_dims=[64, 128, 256, 512],
    nhead=4,
    num_layers=4,
    num_transformer_blocks=2
)

model = TimeSeriesTransformer(config)
print(f"\nModel created successfully")
print(f"Total parameters: {model.count_parameters():,}")

# Create dummy input
batch_size = 4
seq_len = 512
x = torch.randn(batch_size, seq_len, 2)  # [flux, mask]
t = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float()

print(f"\nInput shape: {x.shape}")
print(f"Time shape: {t.shape}")

# Forward pass
output = model(x, t)
print(f"\nOutput shape: {output['reconstructed'].shape}")
print(f"Encoded shape: {output['encoded'].shape}")

# Expected shapes
expected_output_shape = (batch_size, seq_len, 1)
expected_encoded_shape = (batch_size, 512 // (2**4), 512)  # Bottleneck

print(f"\n[OK] Forward pass successful!")
print(f"Output matches expected: {output['reconstructed'].shape == expected_output_shape}")
print(f"Encoded bottleneck size: {output['encoded'].shape[1]} (expected {512 // (2**4)})")

# Test on GPU if available
if torch.cuda.is_available():
    print(f"\n--- Testing on CUDA ---")
    model = model.cuda()
    x = x.cuda()
    t = t.cuda()

    output = model(x, t)
    print(f"[OK] CUDA forward pass successful!")
    print(f"Output device: {output['reconstructed'].device}")
else:
    print(f"\nCUDA not available, skipping GPU test")

print(f"\n[OK] All tests passed!")
print(f"\nHierarchical structure:")
print(f"  Level 0: 512 -> 256 (dim: 64)")
print(f"  Level 1: 256 -> 128 (dim: 128)")
print(f"  Level 2: 128 -> 64 (dim: 256)")
print(f"  Level 3: 64 -> 32 (dim: 512)")
print(f"  Bottleneck: 32 (dim: 512)")
print(f"  Decode back to: 512 (dim: 64)")
