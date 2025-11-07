#!/usr/bin/env python
"""
Evaluate trained models on test data.

Usage:
    python scripts/evaluate.py --checkpoint experiments/results/unet/ps/checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint path/to/model.pt --data data/test_data.pt --visualize
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
from lcgen.evaluation import (
    evaluate_reconstruction,
    plot_reconstruction,
    plot_latent_space
)


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

    print(f"Loaded {model_type} model with {model.count_parameters():,} parameters")

    return model, checkpoint


def evaluate_model(model, data_loader, device, visualize=False, output_dir=None):
    """Evaluate model on dataset"""
    print("\nEvaluating model...")

    all_predictions = []
    all_targets = []
    all_latents = []

    with torch.no_grad():
        for batch in data_loader:
            # Move to device
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            # Forward pass
            output = model(x)
            reconstructed = output['reconstructed']
            encoded = output['encoded']

            # Store results
            all_predictions.append(reconstructed.cpu())
            all_targets.append(x.cpu())

            # Store latent representations
            if hasattr(model, 'get_compact_latent'):
                latent = model.get_compact_latent(x)
                all_latents.append(latent.cpu())
            else:
                # For models without get_compact_latent, use encoded directly
                if encoded.dim() > 2:
                    # Flatten spatial dimensions
                    latent = encoded.flatten(start_dim=1)
                else:
                    latent = encoded
                all_latents.append(latent.cpu())

    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    latents = torch.cat(all_latents, dim=0)

    # Compute metrics
    metrics = evaluate_reconstruction(predictions, targets)

    print("\nEvaluation Metrics:")
    print("=" * 50)
    for metric_name, value in metrics.items():
        print(f"  {metric_name:20s}: {value:.6f}")
    print("=" * 50)

    # Visualization
    if visualize and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating visualizations...")

        # Plot random reconstructions
        num_examples = min(5, len(predictions))
        indices = np.random.choice(len(predictions), num_examples, replace=False)

        for i, idx in enumerate(indices):
            plot_reconstruction(
                predictions[idx].numpy(),
                targets[idx].numpy(),
                title=f"Reconstruction Example {i+1}",
                save_path=output_dir / f"reconstruction_{i+1}.png",
                show=False
            )

        print(f"Saved {num_examples} reconstruction plots to {output_dir}")

        # Plot latent space (if not too high-dimensional)
        if latents.shape[1] <= 512:  # Only if latent dim is reasonable
            try:
                plot_latent_space(
                    latents.numpy(),
                    method='pca',
                    title="Latent Space (PCA)",
                    save_path=output_dir / "latent_space_pca.png",
                    show=False
                )
                print(f"Saved latent space visualization to {output_dir}")
            except Exception as e:
                print(f"Could not generate latent space plot: {e}")

    return metrics, predictions, targets, latents


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained LC-Gen models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to test data (if None, uses data from training)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save visualizations')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)

    # Load data
    if args.data is not None:
        # Load specified test data
        print(f"\nLoading test data from {args.data}")
        test_data = torch.load(args.data, weights_only=False)

        if isinstance(test_data, torch.Tensor):
            dataset = PowerSpectrumDataset(test_data)
        else:
            dataset = test_data
    else:
        # Try to load data from training config
        print("\nNo test data specified, attempting to load from training config...")
        try:
            from train import load_data, create_data_loaders
            from lcgen.utils import Config

            # Reconstruct config
            config_dict = checkpoint['config']
            config = Config(**config_dict)

            # Load and split data
            full_dataset = load_data(config)
            _, _, test_loader = create_data_loaders(full_dataset, config)

            print(f"Loaded test set with {len(test_loader.dataset)} samples")

            # Evaluate
            if args.output_dir is None:
                args.output_dir = Path(checkpoint['config']['output_dir']) / 'evaluation'

            metrics, predictions, targets, latents = evaluate_model(
                model, test_loader, device,
                visualize=args.visualize,
                output_dir=args.output_dir
            )

            print("\nEvaluation complete!")
            return

        except Exception as e:
            print(f"Could not load data from training config: {e}")
            print("Please specify test data with --data argument")
            sys.exit(1)

    # Create data loader
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Test set size: {len(dataset)} samples")

    # Set output directory
    if args.output_dir is None:
        checkpoint_dir = Path(args.checkpoint).parent
        args.output_dir = checkpoint_dir.parent / 'evaluation'

    # Evaluate
    metrics, predictions, targets, latents = evaluate_model(
        model, test_loader, device,
        visualize=args.visualize,
        output_dir=args.output_dir
    )

    # Save results
    if args.output_dir:
        results_file = Path(args.output_dir) / 'evaluation_results.pt'
        torch.save({
            'metrics': metrics,
            'checkpoint_path': args.checkpoint,
        }, results_file)
        print(f"\nSaved evaluation results to {results_file}")

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
