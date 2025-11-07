"""
Evaluate trained MLP autoencoders on test data.

Provides:
1. Quantitative metrics (MSE/MAE) by LC type and sampling strategy
2. Qualitative visualizations (reconstructions, latent space)
3. Comparison across channels (PSD, ACF, F-stat)
"""

import numpy as np
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lcgen.models.mlp import MLPAutoencoder
from lcgen.data.masking import block_predefined_mask


class SpectralDataset(Dataset):
    """Dataset for evaluation (no masking)."""

    def __init__(self, data, metadata=None):
        self.data = torch.from_numpy(data).float()
        self.metadata = metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'idx': idx
        }


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = MLPAutoencoder(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def compute_metrics(model, dataloader, device):
    """Compute reconstruction metrics."""
    model.eval()

    all_mse = []
    all_mae = []
    all_encoded = []
    all_indices = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing metrics"):
            x = batch['data'].to(device)
            idx = batch['idx']

            # Forward pass
            output = model(x)
            x_recon = output['reconstructed']
            encoded = output['encoded']

            # Compute metrics per sample
            mse = ((x - x_recon) ** 2).mean(dim=-1).cpu().numpy()
            mae = torch.abs(x - x_recon).mean(dim=-1).cpu().numpy()

            all_mse.extend(mse)
            all_mae.extend(mae)
            all_encoded.append(encoded.cpu().numpy())
            all_indices.extend(idx.numpy())

    all_encoded = np.concatenate(all_encoded, axis=0)

    return {
        'mse': np.array(all_mse),
        'mae': np.array(all_mae),
        'encoded': all_encoded,
        'indices': np.array(all_indices)
    }


def plot_reconstruction_grid(model, dataset, device, save_path, n_samples=16):
    """Plot grid of reconstruction examples."""
    model.eval()

    n_cols = 4
    n_rows = n_samples // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    # Sample diverse indices
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            batch = dataset[idx]
            x = batch['data'].unsqueeze(0).to(device)

            output = model(x)
            x_recon = output['reconstructed'].cpu().numpy()[0]
            x_orig = x.cpu().numpy()[0]

            ax = axes[i]
            ax.plot(x_orig, 'k-', label='Original', alpha=0.7, linewidth=1)
            ax.plot(x_recon, 'r--', label='Reconstructed', alpha=0.7, linewidth=1)

            # Compute MSE for this sample
            mse = np.mean((x_orig - x_recon) ** 2)
            ax.set_title(f'Sample {idx} | MSE: {mse:.4f}', fontsize=10)
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved reconstruction grid to {save_path}")


def plot_latent_space(encoded, metadata, lc_types, sampling_strategies, save_path):
    """Plot t-SNE of latent space colored by metadata."""

    # Subsample for faster t-SNE (use up to 5000 samples)
    n_samples = min(5000, len(encoded))
    indices = np.random.choice(len(encoded), n_samples, replace=False)

    encoded_subset = encoded[indices]
    lc_types_subset = lc_types[indices]
    sampling_subset = sampling_strategies[indices]

    print(f"Computing t-SNE on {n_samples} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(encoded_subset)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Color by LC type
    ax = axes[0]
    unique_types = np.unique(lc_types_subset)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))

    for i, lc_type in enumerate(unique_types):
        mask = lc_types_subset == lc_type
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                  c=[colors[i]], label=lc_type.decode() if isinstance(lc_type, bytes) else lc_type,
                  alpha=0.6, s=20)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Latent Space by Light Curve Type')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Color by sampling strategy
    ax = axes[1]
    unique_sampling = np.unique(sampling_subset)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sampling)))

    for i, sampling in enumerate(unique_sampling):
        mask = sampling_subset == sampling
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                  c=[colors[i]], label=sampling.decode() if isinstance(sampling, bytes) else sampling,
                  alpha=0.6, s=20)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Latent Space by Sampling Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved latent space plot to {save_path}")


def plot_metrics_by_category(metrics_dict, lc_types, sampling_strategies, save_path):
    """Plot metrics broken down by LC type and sampling strategy."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE by LC type
    ax = axes[0, 0]
    unique_types = np.unique(lc_types)
    mse_by_type = [metrics_dict['mse'][lc_types == t] for t in unique_types]
    labels = [t.decode() if isinstance(t, bytes) else t for t in unique_types]

    ax.boxplot(mse_by_type, labels=labels)
    ax.set_ylabel('MSE')
    ax.set_title('MSE by Light Curve Type')
    ax.grid(True, alpha=0.3)

    # MSE by sampling strategy
    ax = axes[0, 1]
    unique_sampling = np.unique(sampling_strategies)
    mse_by_sampling = [metrics_dict['mse'][sampling_strategies == s] for s in unique_sampling]
    labels = [s.decode() if isinstance(s, bytes) else s for s in unique_sampling]

    ax.boxplot(mse_by_sampling, labels=labels)
    ax.set_ylabel('MSE')
    ax.set_title('MSE by Sampling Strategy')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # MAE by LC type
    ax = axes[1, 0]
    mae_by_type = [metrics_dict['mae'][lc_types == t] for t in unique_types]
    labels = [t.decode() if isinstance(t, bytes) else t for t in unique_types]

    ax.boxplot(mae_by_type, labels=labels)
    ax.set_ylabel('MAE')
    ax.set_title('MAE by Light Curve Type')
    ax.grid(True, alpha=0.3)

    # MAE by sampling strategy
    ax = axes[1, 1]
    mae_by_sampling = [metrics_dict['mae'][sampling_strategies == s] for s in unique_sampling]
    labels = [s.decode() if isinstance(s, bytes) else s for s in unique_sampling]

    ax.boxplot(mae_by_sampling, labels=labels)
    ax.set_ylabel('MAE')
    ax.set_title('MAE by Sampling Strategy')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved metrics breakdown to {save_path}")


def print_summary_stats(metrics_dict, lc_types, sampling_strategies, channel):
    """Print summary statistics."""

    print(f"\n{'='*60}")
    print(f"Summary Statistics - {channel.upper()}")
    print(f"{'='*60}")

    print(f"\nOverall Metrics:")
    print(f"  MSE: {np.mean(metrics_dict['mse']):.6f} ± {np.std(metrics_dict['mse']):.6f}")
    print(f"  MAE: {np.mean(metrics_dict['mae']):.6f} ± {np.std(metrics_dict['mae']):.6f}")

    print(f"\nMetrics by Light Curve Type:")
    unique_types = np.unique(lc_types)
    for lc_type in unique_types:
        mask = lc_types == lc_type
        label = lc_type.decode() if isinstance(lc_type, bytes) else lc_type
        print(f"  {label:15s} - MSE: {np.mean(metrics_dict['mse'][mask]):.6f}, "
              f"MAE: {np.mean(metrics_dict['mae'][mask]):.6f}")

    print(f"\nMetrics by Sampling Strategy:")
    unique_sampling = np.unique(sampling_strategies)
    for sampling in unique_sampling:
        mask = sampling_strategies == sampling
        label = sampling.decode() if isinstance(sampling, bytes) else sampling
        print(f"  {label:20s} - MSE: {np.mean(metrics_dict['mse'][mask]):.6f}, "
              f"MAE: {np.mean(metrics_dict['mae'][mask]):.6f}")


def main(args):
    """Main evaluation function."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    print(f"\nLoading data from {args.input}...")
    with h5py.File(args.input, 'r') as f:
        data = f[args.channel][:]
        lc_types = f['lc_type'][:]
        sampling_strategies = f['sampling_strategy'][:]

        print(f"Loaded {args.channel}: {data.shape}")
        print(f"LC types: {len(np.unique(lc_types))} unique")
        print(f"Sampling strategies: {len(np.unique(sampling_strategies))} unique")

    # Create dataset
    dataset = SpectralDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Load model
    checkpoint_path = Path(args.checkpoint)
    print(f"\nLoading model from {checkpoint_path}...")
    model, checkpoint = load_model(checkpoint_path, device)

    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Best validation loss: {checkpoint.get('val_loss', 'N/A')}")

    # Compute metrics
    print(f"\nEvaluating on {len(dataset)} samples...")
    metrics = compute_metrics(model, dataloader, device)

    # Print summary statistics
    print_summary_stats(metrics, lc_types, sampling_strategies, args.channel)

    # Generate visualizations
    print(f"\nGenerating visualizations...")

    # Reconstruction grid
    plot_reconstruction_grid(
        model, dataset, device,
        output_dir / f'mlp_{args.channel}_reconstruction_grid.png',
        n_samples=16
    )

    # Latent space visualization
    plot_latent_space(
        metrics['encoded'],
        metrics,
        lc_types,
        sampling_strategies,
        output_dir / f'mlp_{args.channel}_latent_space.png'
    )

    # Metrics breakdown
    plot_metrics_by_category(
        metrics,
        lc_types,
        sampling_strategies,
        output_dir / f'mlp_{args.channel}_metrics_breakdown.png'
    )

    # Save metrics to CSV
    csv_path = output_dir / f'mlp_{args.channel}_metrics.csv'
    df = pd.DataFrame({
        'mse': metrics['mse'],
        'mae': metrics['mae'],
        'lc_type': [lt.decode() if isinstance(lt, bytes) else lt for lt in lc_types],
        'sampling_strategy': [s.decode() if isinstance(s, bytes) else s for s in sampling_strategies]
    })
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")

    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained MLP autoencoder"
    )

    parser.add_argument(
        '--input', type=str,
        default='data/mock_lightcurves/preprocessed.h5',
        help='Input HDF5 file with preprocessed features'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--channel', type=str, required=True,
        choices=['psd', 'acf', 'fstat'],
        help='Which channel the model was trained on'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results/mlp',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='Batch size for evaluation'
    )

    args = parser.parse_args()
    main(args)
