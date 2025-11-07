#!/usr/bin/env python
"""
Export latent embeddings from trained models.

Usage:
    python scripts/export_embeddings.py --checkpoint path/to/model.pt --data data.pt --output embeddings.npy
    python scripts/export_embeddings.py --checkpoint best_model.pt --data data.pt --format pt
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lcgen.models import (
    PowerSpectrumUNetAutoencoder, UNetConfig,
    MLPAutoencoder, MLPConfig,
    TimeSeriesTransformer, TransformerConfig
)
from lcgen.data import PowerSpectrumDataset


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model configuration
    config = checkpoint['config']
    model_type = config['model']['model_type']

    # Create model
    if model_type == 'unet':
        model_config = UNetConfig(**config['model'])
        model = PowerSpectrumUNetAutoencoder(model_config)
    elif model_type == 'mlp':
        model_config = MLPConfig(**config['model'])
        model = MLPAutoencoder(model_config)
    elif model_type == 'transformer':
        model_config = TransformerConfig(**config['model'])
        model = TimeSeriesTransformer(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded {model_type} model")
    return model


def extract_embeddings(model, data_loader, device, flatten=True):
    """Extract latent embeddings from all data"""
    print("\nExtracting embeddings...")

    all_embeddings = []
    all_labels = []  # Store indices for later reference

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move to device
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            # Encode
            if hasattr(model, 'get_compact_latent'):
                # For UNet models, get compact latent (global pooled)
                embeddings = model.get_compact_latent(x)
            else:
                # For other models, use encode method
                embeddings = model.encode(x)

                # Flatten if requested and if multi-dimensional
                if flatten and embeddings.dim() > 2:
                    embeddings = embeddings.flatten(start_dim=1)

            all_embeddings.append(embeddings.cpu())

            # Track batch indices
            batch_size = x.shape[0]
            start_idx = batch_idx * data_loader.batch_size
            batch_labels = list(range(start_idx, start_idx + batch_size))
            all_labels.extend(batch_labels)

    # Concatenate all embeddings
    embeddings = torch.cat(all_embeddings, dim=0)

    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings, all_labels


def main():
    parser = argparse.ArgumentParser(description='Export latent embeddings from trained models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save embeddings')
    parser.add_argument('--format', type=str, default='npy', choices=['npy', 'pt', 'csv'],
                        help='Output format (npy, pt, or csv)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--no-flatten', action='store_true',
                        help='Do not flatten multi-dimensional embeddings')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device)

    # Load data
    print(f"\nLoading data from {args.data}")
    data = torch.load(args.data, weights_only=False)

    if isinstance(data, torch.Tensor):
        dataset = PowerSpectrumDataset(data)
    else:
        dataset = data

    print(f"Loaded dataset with {len(dataset)} samples")

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Important: don't shuffle for consistent ordering
        num_workers=0
    )

    # Extract embeddings
    embeddings, labels = extract_embeddings(
        model, data_loader, device,
        flatten=not args.no_flatten
    )

    # Convert to numpy
    embeddings_np = embeddings.numpy()

    # Save in requested format
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == 'npy':
        np.save(output_path, embeddings_np)
        print(f"\nSaved embeddings to {output_path} (NumPy format)")

    elif args.format == 'pt':
        torch.save({
            'embeddings': embeddings,
            'labels': labels,
            'checkpoint': args.checkpoint
        }, output_path)
        print(f"\nSaved embeddings to {output_path} (PyTorch format)")

    elif args.format == 'csv':
        import pandas as pd

        # Create DataFrame
        df = pd.DataFrame(
            embeddings_np,
            columns=[f'dim_{i}' for i in range(embeddings_np.shape[1])]
        )
        df.insert(0, 'index', labels)

        df.to_csv(output_path, index=False)
        print(f"\nSaved embeddings to {output_path} (CSV format)")

    # Print summary statistics
    print("\nEmbedding Statistics:")
    print(f"  Shape: {embeddings_np.shape}")
    print(f"  Mean: {embeddings_np.mean():.6f}")
    print(f"  Std: {embeddings_np.std():.6f}")
    print(f"  Min: {embeddings_np.min():.6f}")
    print(f"  Max: {embeddings_np.max():.6f}")

    print("\nEmbedding export complete!")


if __name__ == '__main__':
    main()
