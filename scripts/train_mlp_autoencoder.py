"""
Train MLP autoencoder on preprocessed multi-modal features.

Supports training on individual channels (PSD, ACF, F-stat) with block masking.
"""

import numpy as np
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lcgen.models.mlp import MLPAutoencoder, MLPConfig
from lcgen.data.masking import dynamic_block_mask


class SpectralDataset(Dataset):
    """Dataset for single-channel spectral data (PSD, ACF, or F-stat)."""

    def __init__(self, data, min_block_size=1, max_block_size=None,
                 min_mask_ratio=0.1, max_mask_ratio=0.9):
        """
        Args:
            data: Numpy array of shape (N, n_bins)
            min_block_size: Minimum block size for dynamic masking
            max_block_size: Maximum block size (default: n_bins // 2)
            min_mask_ratio: Minimum masking fraction
            max_mask_ratio: Maximum masking fraction
        """
        self.data = torch.from_numpy(data).float()
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        # Apply dynamic block masking (random block size and mask ratio each time)
        x_masked, mask, block_size, mask_ratio = dynamic_block_mask(
            x,
            min_block_size=self.min_block_size,
            max_block_size=self.max_block_size,
            min_mask_ratio=self.min_mask_ratio,
            max_mask_ratio=self.max_mask_ratio
        )

        return {
            'masked': x_masked,
            'target': x,
            'mask': mask,
            'block_size': block_size,
            'mask_ratio': mask_ratio
        }


def load_preprocessed_data(hdf5_file, channel='psd'):
    """
    Load preprocessed data from HDF5 file.

    Args:
        hdf5_file: Path to HDF5 file
        channel: Which channel to load ('psd', 'acf', 'fstat')

    Returns:
        data: Numpy array of shape (N, n_bins)
    """
    with h5py.File(hdf5_file, 'r') as f:
        data = f[channel][:]
        print(f"Loaded {channel}: {data.shape}")

        # Check for NaNs
        n_nans = np.sum(np.isnan(data))
        if n_nans > 0:
            print(f"Warning: {n_nans} NaN values found in {channel}")
            # Replace NaNs with 0 (should not happen with current preprocessing)
            data = np.nan_to_num(data, nan=0.0)

        return data


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_masked_loss = 0
    total_unmasked_loss = 0
    n_batches = 0

    for batch in dataloader:
        x_masked = batch['masked'].to(device)
        x_target = batch['target'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(x_masked)
        x_recon = output['reconstructed']

        # Compute loss over ALL regions (both masked and unmasked)
        loss = criterion(x_recon, x_target)

        # Also track masked vs unmasked separately for monitoring
        masked_loss = criterion(x_recon[mask], x_target[mask])
        unmasked_loss = criterion(x_recon[~mask], x_target[~mask])

        # Backprop and optimize on full reconstruction loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_masked_loss += masked_loss.item()
        total_unmasked_loss += unmasked_loss.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'masked_loss': total_masked_loss / n_batches,
        'unmasked_loss': total_unmasked_loss / n_batches
    }


def validate(model, dataloader, criterion, device):
    """Validate on validation set."""
    model.eval()
    total_loss = 0
    total_masked_loss = 0
    total_unmasked_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x_masked = batch['masked'].to(device)
            x_target = batch['target'].to(device)
            mask = batch['mask'].to(device)

            # Forward pass
            output = model(x_masked)
            x_recon = output['reconstructed']

            # Compute loss over ALL regions
            loss = criterion(x_recon, x_target)

            # Also track masked vs unmasked separately for monitoring
            masked_loss = criterion(x_recon[mask], x_target[mask])
            unmasked_loss = criterion(x_recon[~mask], x_target[~mask])

            total_loss += loss.item()
            total_masked_loss += masked_loss.item()
            total_unmasked_loss += unmasked_loss.item()
            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'masked_loss': total_masked_loss / n_batches,
        'unmasked_loss': total_unmasked_loss / n_batches
    }


def plot_reconstruction_samples(model, dataset, device, save_path, n_samples=4):
    """Plot sample reconstructions."""
    model.eval()

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            batch = dataset[idx]
            x_masked = batch['masked'].unsqueeze(0).to(device)
            x_target = batch['target'].cpu().numpy()
            mask = batch['mask'].cpu().numpy()

            output = model(x_masked)
            x_recon = output['reconstructed'].cpu().numpy()[0]

            ax = axes[i]
            ax.plot(x_target, 'k-', label='Target', alpha=0.7, linewidth=1)
            ax.plot(x_recon, 'r-', label='Reconstructed', alpha=0.7, linewidth=1)

            # Highlight masked regions
            masked_indices = np.where(mask)[0]
            if len(masked_indices) > 0:
                ax.scatter(masked_indices, x_recon[masked_indices],
                          c='red', s=10, alpha=0.5, label='Masked regions')

            ax.legend()
            ax.set_xlabel('Bin')
            ax.set_ylabel('Value')
            ax.set_title(f'Sample {idx}')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved reconstruction plot to {save_path}")


def main(args):
    """Main training loop."""

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {args.input}...")
    data = load_preprocessed_data(args.input, channel=args.channel)
    n_bins = data.shape[1]

    print(f"\nDataset statistics:")
    print(f"  Shape: {data.shape}")
    print(f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    print(f"  Mean: {np.mean(data):.3f}")
    print(f"  Std: {np.std(data):.3f}")

    # Create dataset with dynamic masking
    dataset = SpectralDataset(
        data,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        min_mask_ratio=args.min_mask_ratio,
        max_mask_ratio=args.max_mask_ratio
    )

    # Train/val split
    n_train = int(len(dataset) * args.train_split)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"\nDataset split:")
    print(f"  Train: {n_train} samples")
    print(f"  Val: {n_val} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create model
    print(f"\nCreating MLP autoencoder...")
    config = MLPConfig(
        input_dim=n_bins,
        encoder_hidden_dims=args.encoder_dims,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        activation=args.activation
    )

    model = MLPAutoencoder(config).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    max_block_display = args.max_block_size if args.max_block_size else "seq_len//2"
    print(f"Dynamic masking: block size [{args.min_block_size}, {max_block_display}], "
          f"mask ratio [{args.min_mask_ratio:.0%}, {args.max_mask_ratio:.0%}]")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_metrics['loss'])

        # Track losses
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.6f} | "
              f"Masked: {train_metrics['masked_loss']:.6f} | "
              f"Unmasked: {train_metrics['unmasked_loss']:.6f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.6f} | "
              f"Masked: {val_metrics['masked_loss']:.6f} | "
              f"Unmasked: {val_metrics['unmasked_loss']:.6f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = output_dir / f'mlp_{args.channel}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'config': config
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")

        # Plot samples every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            plot_path = output_dir / f'mlp_{args.channel}_recon_epoch{epoch+1}.png'
            plot_reconstruction_samples(model, val_dataset, device, plot_path)

    # Save final model
    final_path = output_dir / f'mlp_{args.channel}_final.pt'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, final_path)
    print(f"\nSaved final model to {final_path}")

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'MLP Autoencoder Training - {args.channel.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'mlp_{args.channel}_training_curve.png', dpi=150)
    plt.close()

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MLP autoencoder on spectral features"
    )

    # Data arguments
    parser.add_argument(
        '--input', type=str,
        default='data/mock_lightcurves/preprocessed.h5',
        help='Input HDF5 file with preprocessed features'
    )
    parser.add_argument(
        '--channel', type=str, default='psd',
        choices=['psd', 'acf', 'fstat'],
        help='Which channel to train on'
    )
    parser.add_argument(
        '--output_dir', type=str, default='models/mlp',
        help='Output directory for models and plots'
    )

    # Model arguments
    parser.add_argument(
        '--encoder_dims', type=int, nargs='+',
        default=[512, 256, 128],
        help='Encoder hidden dimensions'
    )
    parser.add_argument(
        '--latent_dim', type=int, default=64,
        help='Latent dimension'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.0,
        help='Dropout rate'
    )
    parser.add_argument(
        '--activation', type=str, default='gelu',
        choices=['relu', 'gelu', 'silu', 'tanh', 'leakyrelu'],
        help='Activation function'
    )

    # Training arguments
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-5,
        help='Weight decay for AdamW'
    )
    parser.add_argument(
        '--train_split', type=float, default=0.9,
        help='Fraction of data for training'
    )

    # Dynamic masking arguments
    parser.add_argument(
        '--min_block_size', type=int, default=1,
        help='Minimum block size for dynamic masking'
    )
    parser.add_argument(
        '--max_block_size', type=int, default=None,
        help='Maximum block size (default: seq_len // 2)'
    )
    parser.add_argument(
        '--min_mask_ratio', type=float, default=0.1,
        help='Minimum masking fraction'
    )
    parser.add_argument(
        '--max_mask_ratio', type=float, default=0.9,
        help='Maximum masking fraction'
    )

    # Other arguments
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    main(args)
