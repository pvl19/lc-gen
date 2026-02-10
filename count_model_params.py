"""Count parameters in BiDirectionalMinGRU model with and without conv channels."""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lcgen.models.simple_min_gru import BiDirectionalMinGRU


def count_parameters(model, name="Model"):
    """Count trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{name}:")
    print(f"  Total parameters: {total:,}")

    # Breakdown by component
    if hasattr(model, 'forward_cell'):
        rnn_params = sum(p.numel() for p in model.forward_cell.parameters())
        if model.direction == 'bi':
            rnn_params += sum(p.numel() for p in model.backward_cell.parameters())
        print(f"  RNN cells: {rnn_params:,}")

    if hasattr(model, 'meta_encoder') and model.meta_encoder is not None:
        meta_params = sum(p.numel() for p in model.meta_encoder.parameters())
        print(f"  Metadata encoder: {meta_params:,}")

    if hasattr(model, 'ps_encoder') and model.ps_encoder is not None:
        ps_params = sum(p.numel() for p in model.ps_encoder.parameters())
        acf_params = sum(p.numel() for p in model.acf_encoder.parameters())
        ps_proj_params = sum(p.numel() for p in model.ps_conv_proj.parameters())
        acf_proj_params = sum(p.numel() for p in model.acf_conv_proj.parameters())
        conv_total = ps_params + acf_params + ps_proj_params + acf_proj_params
        print(f"  Conv channels (PS+ACF encoders + projections): {conv_total:,}")
        print(f"    - PS encoder: {ps_params:,}")
        print(f"    - ACF encoder: {acf_params:,}")
        print(f"    - PS projection: {ps_proj_params:,}")
        print(f"    - ACF projection: {acf_proj_params:,}")

    if hasattr(model, 'gauss_head'):
        head_params = sum(p.numel() for p in model.gauss_head.parameters())
        print(f"  Prediction head: {head_params:,}")

    return total


if __name__ == '__main__':
    print("="*60)
    print("Model Parameter Count Analysis")
    print("="*60)

    # Configuration
    hidden_size = 64
    direction = 'bi'
    num_meta_features = 8

    # Model WITHOUT conv channels (baseline)
    model_base = BiDirectionalMinGRU(
        hidden_size=hidden_size,
        direction=direction,
        use_flow=False,
        num_meta_features=num_meta_features,
        use_conv_channels=False
    )

    base_params = count_parameters(model_base, "Baseline Model (no conv channels)")

    # Model WITH conv channels
    conv_config = {
        'input_length': 16000,
        'encoder_dims': [4, 8, 16, 32],
        'num_layers': 4,
        'activation': 'gelu',
        'in_channels': 1
    }

    model_conv = BiDirectionalMinGRU(
        hidden_size=hidden_size,
        direction=direction,
        use_flow=False,
        num_meta_features=num_meta_features,
        use_conv_channels=True,
        conv_config=conv_config
    )

    conv_params = count_parameters(model_conv, "Model WITH conv channels")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline model: {base_params:,} parameters")
    print(f"Model with conv: {conv_params:,} parameters")
    print(f"Increase: {conv_params - base_params:,} parameters ({100*(conv_params - base_params)/base_params:.2f}%)")
    print("="*60)
