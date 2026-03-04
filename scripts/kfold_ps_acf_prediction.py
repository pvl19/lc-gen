"""K-fold cross-validation for predicting power spectrum and ACF from latent space.

This script trains k separate models, each on (k-1)/k of the data, and generates
predictions for the held-out fold. The result is an unbiased prediction
for every sample, made by a model that never saw that sample during training.

Usage:
    python scripts/kfold_ps_acf_prediction.py \
        --model_path output/simple_rnn/models/model.pt \
        --output_dir output/ps_acf_predictor/kfold \
        --n_folds 10
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Ensure local 'src' is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class PSACFPredictor(nn.Module):
    """MLP to predict power spectrum and ACF from latent vectors."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [256, 512, 1024], dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def load_data(model_path: str, h5_path: str, pickle_path: str,
              hidden_size: int, direction: str, mode: str, use_flow: bool,
              max_length: int, batch_size: int, pooling_mode: str,
              max_samples: int = None, use_metadata: bool = False,
              use_conv_channels: bool = False, conv_config: dict = None,
              target_length: int = 16000, device: torch.device = None):
    """Extract latent vectors and corresponding PS/ACF from the model."""
    from plot_umap_latent import load_model, extract_latent_vectors

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load PS and ACF from h5 file
    with h5py.File(h5_path, 'r') as f:
        h5_size = len(f['flux'])

        if 'power' not in f or 'acf' not in f:
            raise ValueError(f"H5 file must contain 'power' and 'acf' datasets. Found: {list(f.keys())}")

        power_spectrum = f['power'][:]
        acf = f['acf'][:]

    print(f"Loaded power spectrum shape: {power_spectrum.shape}")
    print(f"Loaded ACF shape: {acf.shape}")

    # Determine sample indices
    if max_samples is not None:
        sample_indices = np.arange(min(max_samples, h5_size))
    else:
        sample_indices = np.arange(h5_size)

    # Determine number of metadata features
    num_meta_features = 8 if use_metadata else 0

    # Load model and extract latents
    model = load_model(model_path, hidden_size, direction, mode, use_flow, device,
                       num_meta_features=num_meta_features,
                       use_conv_channels=use_conv_channels,
                       conv_config=conv_config)
    latent_vectors = extract_latent_vectors(
        model, h5_path, device, max_length, batch_size, pooling_mode, sample_indices,
        use_metadata=use_metadata, use_conv_channels=use_conv_channels
    )

    # Get corresponding PS and ACF
    power_spectrum = power_spectrum[sample_indices]
    acf = acf[sample_indices]

    # Truncate or pad to target length if needed
    if power_spectrum.shape[1] != target_length:
        if power_spectrum.shape[1] > target_length:
            power_spectrum = power_spectrum[:, :target_length]
            acf = acf[:, :target_length]
        else:
            # Pad with zeros
            pad_width = target_length - power_spectrum.shape[1]
            power_spectrum = np.pad(power_spectrum, ((0, 0), (0, pad_width)), mode='constant')
            acf = np.pad(acf, ((0, 0), (0, pad_width)), mode='constant')

    print(f"Extracted {len(latent_vectors)} samples")
    print(f"Latent dimension: {latent_vectors.shape[1]}")
    print(f"Target PS/ACF length: {target_length}")

    return latent_vectors, power_spectrum, acf, sample_indices


def train_single_fold(X_train, y_train, X_val, y_val,
                      hidden_dims, dropout, lr, weight_decay, n_epochs, batch_size, device,
                      output_dim):
    """Train a single model on training data, return predictions on validation data."""

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = PSACFPredictor(X_train.shape[1], output_dim, hidden_dims, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_train_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        train_loss /= len(X_train)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

    # Restore best model and predict on validation set
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        val_pred = model(X_val_t).cpu().numpy()

    return val_pred, best_train_loss, best_state


def run_kfold_cv(latent_vectors, targets, target_name,
                 n_folds=10, hidden_dims=[256, 512, 1024], dropout=0.2,
                 lr=1e-3, weight_decay=1e-4, n_epochs=50, batch_size=64,
                 device=None, seed=42, save_models_dir=None):
    """Run k-fold cross-validation for a single target (PS or ACF)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(seed)
    n_samples = len(latent_vectors)
    output_dim = targets.shape[1]

    # Normalize inputs
    X_mean, X_std = latent_vectors.mean(axis=0), latent_vectors.std(axis=0) + 1e-8
    X_normalized = (latent_vectors - X_mean) / X_std

    # Normalize targets (per-feature normalization)
    y_mean, y_std = targets.mean(axis=0), targets.std(axis=0) + 1e-8
    y_normalized = (targets - y_mean) / y_std

    # Store normalization params
    normalization_params = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'latent_dim': latent_vectors.shape[1],
        'output_dim': output_dim,
    }

    # Shuffle indices and create folds
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_folds

    # Store predictions for each sample
    all_predictions = np.zeros_like(targets)
    fold_assignments = np.zeros(n_samples, dtype=int)
    fold_losses = []
    fold_models = []

    print(f"\nRunning {n_folds}-fold cross-validation for {target_name} on {n_samples} samples...")
    print(f"Output dimension: {output_dim}")
    print(f"Each fold: ~{fold_size} validation samples, ~{n_samples - fold_size} training samples")

    for fold in range(n_folds):
        # Define validation indices for this fold
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        # Get train/val data
        X_train, y_train = X_normalized[train_idx], y_normalized[train_idx]
        X_val, y_val = X_normalized[val_idx], y_normalized[val_idx]

        # Train and predict
        val_pred_normalized, train_loss, model_state = train_single_fold(
            X_train, y_train, X_val, y_val,
            hidden_dims, dropout, lr, weight_decay, n_epochs, batch_size, device,
            output_dim
        )

        fold_models.append(model_state)

        # Denormalize predictions
        val_pred = val_pred_normalized * y_std + y_mean

        # Store predictions
        all_predictions[val_idx] = val_pred
        fold_assignments[val_idx] = fold
        fold_losses.append(train_loss)

        # Compute validation metrics for this fold
        val_mse = np.mean((val_pred - targets[val_idx]) ** 2)
        val_mae = np.mean(np.abs(val_pred - targets[val_idx]))

        print(f"  Fold {fold+1}/{n_folds}: train_loss={train_loss:.6f}, "
              f"val_MSE={val_mse:.6f}, val_MAE={val_mae:.6f}")

    # Save models if directory provided
    if save_models_dir is not None:
        save_dir = Path(save_models_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save all fold models and normalization params
        save_dict = {
            'fold_models': fold_models,
            'normalization_params': {k: v.tolist() if isinstance(v, np.ndarray) else v
                                     for k, v in normalization_params.items()},
            'hidden_dims': hidden_dims,
            'dropout': dropout,
            'n_folds': n_folds,
            'seed': seed,
            'fold_indices': indices.tolist(),
        }
        model_path = save_dir / f'kfold_models_{target_name}.pt'
        torch.save(save_dict, model_path)
        print(f"\nSaved {n_folds} fold models to {model_path}")

    return all_predictions, fold_assignments, fold_losses, normalization_params


def plot_results(predictions, targets, target_name, fold_assignments, output_dir, sample_indices=None):
    """Create plots for prediction results."""

    # Compute overall metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    # Per-frequency metrics
    mse_per_freq = np.mean((predictions - targets) ** 2, axis=0)
    mae_per_freq = np.mean(np.abs(predictions - targets), axis=0)

    # Correlation per sample
    correlations = []
    for i in range(len(predictions)):
        corr = np.corrcoef(predictions[i], targets[i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    mean_corr = np.mean(correlations)

    print(f"\n=== K-Fold Results for {target_name} ===")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Mean correlation per sample: {mean_corr:.4f}")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. MSE per frequency bin
    ax1 = axes[0, 0]
    ax1.plot(mse_per_freq, alpha=0.7)
    ax1.set_xlabel('Frequency bin')
    ax1.set_ylabel('MSE')
    ax1.set_title(f'{target_name}: MSE per frequency bin')
    ax1.set_yscale('log')

    # 2. Sample comparison (random samples)
    ax2 = axes[0, 1]
    n_examples = 5
    example_indices = np.random.choice(len(predictions), n_examples, replace=False)
    for i, idx in enumerate(example_indices):
        ax2.plot(targets[idx], alpha=0.5, label=f'True {i+1}')
        ax2.plot(predictions[idx], '--', alpha=0.5, label=f'Pred {i+1}')
    ax2.set_xlabel('Frequency bin')
    ax2.set_ylabel('Value')
    ax2.set_title(f'{target_name}: Example predictions vs true')
    ax2.legend(ncol=2, fontsize=8)

    # 3. Correlation histogram
    ax3 = axes[1, 0]
    ax3.hist(correlations, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(mean_corr, color='r', linestyle='--', lw=2, label=f'Mean={mean_corr:.3f}')
    ax3.set_xlabel('Correlation')
    ax3.set_ylabel('Count')
    ax3.set_title(f'{target_name}: Per-sample correlation distribution')
    ax3.legend()

    # 4. Predicted vs true (aggregated across all frequencies)
    ax4 = axes[1, 1]
    # Sample a subset for plotting
    n_plot = min(10000, predictions.size)
    plot_idx = np.random.choice(predictions.size, n_plot, replace=False)
    flat_pred = predictions.flatten()[plot_idx]
    flat_true = targets.flatten()[plot_idx]
    ax4.scatter(flat_true, flat_pred, alpha=0.1, s=1)
    lims = [min(flat_true.min(), flat_pred.min()), max(flat_true.max(), flat_pred.max())]
    ax4.plot(lims, lims, 'r--', lw=2, label='Perfect')
    ax4.set_xlabel('True value')
    ax4.set_ylabel('Predicted value')
    ax4.set_title(f'{target_name}: Predicted vs True (sampled points)')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'kfold_results_{target_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_dir / f'kfold_results_{target_name}.png'}")

    # Save predictions as numpy array for interactive analysis
    np.save(output_dir / f'{target_name}_predictions.npy', predictions)
    print(f"Saved predictions to {output_dir / f'{target_name}_predictions.npy'}")

    # Save metrics
    metrics = {
        'n_samples': len(targets),
        'output_dim': targets.shape[1],
        'mse': float(mse),
        'mae': float(mae),
        'mean_correlation': float(mean_corr),
    }
    with open(output_dir / f'kfold_metrics_{target_name}.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='K-fold cross-validation for PS/ACF prediction')

    # Data sources
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained BiDirectionalMinGRU model')
    parser.add_argument('--h5_path', type=str, default='data/timeseries_x.h5', help='Path to H5 data file')
    parser.add_argument('--pickle_path', type=str, default='data/star_sector_lc_formatted_with_extra_channels.pickle', help='Path to pickle file')
    parser.add_argument('--output_dir', type=str, default='output/ps_acf_predictor/kfold', help='Output directory')

    # Model extraction settings
    parser.add_argument('--hidden_size', type=int, default=64, help='RNN hidden size')
    parser.add_argument('--direction', type=str, default='bi', choices=['forward', 'backward', 'bi'])
    parser.add_argument('--mode', type=str, default='parallel', choices=['sequential', 'parallel'])
    parser.add_argument('--use_flow', action='store_true', help='Whether RNN uses flow head')
    parser.add_argument('--use_metadata', action='store_true', help='Use stellar metadata as model input')
    parser.add_argument('--use_conv_channels', action='store_true', help='Use convolutional encoders')
    parser.add_argument('--conv_encoder_type', type=str, default='unet', choices=['lightweight', 'unet'])
    parser.add_argument('--max_length', type=int, default=2048, help='Max sequence length')
    parser.add_argument('--pooling_mode', type=str, default='multiscale', choices=['mean', 'multiscale', 'final'])

    # Data selection
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to use')
    parser.add_argument('--target_length', type=int, default=16000, help='Target PS/ACF length')

    # Target selection
    parser.add_argument('--predict_ps', action='store_true', help='Predict power spectrum')
    parser.add_argument('--predict_acf', action='store_true', help='Predict ACF')

    # K-fold settings
    parser.add_argument('--n_folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # MLP architecture
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 512, 1024], help='MLP hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')

    # Training
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    # Default to predicting both if neither specified
    if not args.predict_ps and not args.predict_acf:
        args.predict_ps = True
        args.predict_acf = True

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build conv_config if using convolutional channels
    conv_config = None
    if args.use_conv_channels:
        if args.conv_encoder_type == 'lightweight':
            conv_config = {
                'encoder_type': 'lightweight',
                'hidden_channels': 16,
                'num_layers': 3,
                'activation': 'gelu'
            }
        else:  # unet
            conv_config = {
                'encoder_type': 'unet',
                'input_length': 16000,
                'encoder_dims': [4, 8, 16, 32],
                'num_layers': 4,
                'activation': 'gelu'
            }

    # Extract latent vectors and targets
    print("\n=== Extracting latent vectors and targets ===")
    latent_vectors, power_spectrum, acf, sample_indices = load_data(
        model_path=args.model_path,
        h5_path=args.h5_path,
        pickle_path=args.pickle_path,
        hidden_size=args.hidden_size,
        direction=args.direction,
        mode=args.mode,
        use_flow=args.use_flow,
        max_length=args.max_length,
        batch_size=32,
        pooling_mode=args.pooling_mode,
        max_samples=args.max_samples,
        use_metadata=args.use_metadata,
        use_conv_channels=args.use_conv_channels,
        conv_config=conv_config,
        target_length=args.target_length,
        device=device,
    )

    all_metrics = {}

    # Run k-fold for power spectrum
    if args.predict_ps:
        print("\n=== Running K-Fold for Power Spectrum ===")
        ps_predictions, ps_folds, ps_losses, ps_norm = run_kfold_cv(
            latent_vectors=latent_vectors,
            targets=power_spectrum,
            target_name='power_spectrum',
            n_folds=args.n_folds,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            save_models_dir=output_dir,
        )
        ps_metrics = plot_results(ps_predictions, power_spectrum, 'power_spectrum', ps_folds, output_dir, sample_indices)
        all_metrics['power_spectrum'] = ps_metrics

    # Run k-fold for ACF
    if args.predict_acf:
        print("\n=== Running K-Fold for ACF ===")
        acf_predictions, acf_folds, acf_losses, acf_norm = run_kfold_cv(
            latent_vectors=latent_vectors,
            targets=acf,
            target_name='acf',
            n_folds=args.n_folds,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            save_models_dir=output_dir,
        )
        acf_metrics = plot_results(acf_predictions, acf, 'acf', acf_folds, output_dir, sample_indices)
        all_metrics['acf'] = acf_metrics

    # Save config
    config = vars(args)
    config['latent_dim'] = latent_vectors.shape[1]
    config['n_samples'] = len(latent_vectors)
    config['metrics'] = all_metrics
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\nDone!")


if __name__ == '__main__':
    main()
