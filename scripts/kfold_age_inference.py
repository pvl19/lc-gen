"""K-fold cross-validation for age prediction from latent space + BPRP0.

This script trains k separate models, each on (k-1)/k of the data, and generates
age predictions for the held-out fold. The result is an unbiased age prediction
for every star, made by a model that never saw that star during training.

Usage:
    python scripts/kfold_age_inference.py \
        --model_path output/simple_rnn/models/baseline_v15_masking_real_60e.pt \
        --output_dir output/age_predictor/kfold \
        --n_folds 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Ensure local 'src' is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class AgePredictor(nn.Module):
    """Simple MLP to predict log(age) from latent vectors + BPRP0."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], dropout: float = 0.2):
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
        
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


def load_data_fresh(model_path: str, h5_path: str, pickle_path: str, csv_path: str,
                    hidden_size: int, direction: str, mode: str, use_flow: bool,
                    max_length: int, batch_size: int, pooling_mode: str,
                    max_samples: int = None, stratify_by_age: bool = False,
                    n_age_bins: int = 20, bprp0_min: float = None, bprp0_max: float = None,
                    device: torch.device = None):
    """Extract latent vectors from the model."""
    from plot_umap_latent import (
        load_model, extract_latent_vectors, load_ages, get_stratified_indices
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine which samples to use
    sample_indices = None
    all_ages, all_bprp0, _ = load_ages(pickle_path, csv_path, sample_indices=None)
    
    # Apply BPRP0 cuts
    valid_mask = ~np.isnan(all_ages) & ~np.isnan(all_bprp0)
    if bprp0_min is not None:
        valid_mask &= (all_bprp0 >= bprp0_min)
    if bprp0_max is not None:
        valid_mask &= (all_bprp0 <= bprp0_max)
    
    if max_samples is not None:
        if stratify_by_age:
            ages_for_stratify = np.where(valid_mask, all_ages, np.nan)
            sample_indices = get_stratified_indices(ages_for_stratify, max_samples, n_age_bins)
        else:
            valid_indices = np.where(valid_mask)[0]
            sample_indices = valid_indices[:max_samples]
    else:
        sample_indices = np.where(valid_mask)[0]
    
    # Load model and extract latents
    model = load_model(model_path, hidden_size, direction, mode, use_flow, device)
    latent_vectors = extract_latent_vectors(
        model, h5_path, device, max_length, batch_size, pooling_mode, sample_indices
    )
    
    # Load ages and BPRP0 for selected samples
    ages, bprp0, tic_ids = load_ages(pickle_path, csv_path, sample_indices)
    
    # Filter to valid ages and BPRP0
    valid_mask = ~np.isnan(ages) & ~np.isnan(bprp0)
    latent_vectors = latent_vectors[valid_mask]
    ages = ages[valid_mask]
    bprp0 = bprp0[valid_mask]
    tic_ids = tic_ids[valid_mask]
    
    print(f"Extracted {len(latent_vectors)} samples with valid ages and BPRP0")
    print(f"Latent dimension: {latent_vectors.shape[1]}")
    print(f"Age range: {ages.min():.1f} - {ages.max():.1f} Myr")
    print(f"BPRP0 range: {bprp0.min():.2f} - {bprp0.max():.2f}")
    
    return latent_vectors, ages, bprp0, tic_ids


def train_single_fold(X_train, y_train, X_val, y_val, 
                      hidden_dims, dropout, lr, weight_decay, n_epochs, batch_size, device):
    """Train a single model on training data, return predictions on validation data."""
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=device),
        torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = AgePredictor(X_train.shape[1], hidden_dims, dropout).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    best_train_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
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
        val_pred = model(X_val_t).cpu().numpy().squeeze()
    
    return val_pred, best_train_loss


def run_kfold_cv(latent_vectors, ages, bprp0, tic_ids,
                 n_folds=10, hidden_dims=[128, 64, 32], dropout=0.2,
                 lr=1e-3, weight_decay=1e-4, n_epochs=100, batch_size=64,
                 device=None, seed=42):
    """Run k-fold cross-validation.
    
    Returns:
        predictions: array of predicted ages for all samples
        true_ages: array of true ages
        tic_ids: array of TIC IDs
        fold_indices: which fold each sample was in
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(seed)
    n_samples = len(ages)
    
    # Prepare features
    bprp0_mean, bprp0_std = bprp0.mean(), bprp0.std()
    bprp0_normalized = (bprp0 - bprp0_mean) / (bprp0_std + 1e-8)
    X = np.concatenate([latent_vectors, bprp0_normalized[:, None]], axis=1)
    y = np.log10(ages)  # Predict in log space
    
    # Global normalization (compute on all data - this is fine since it's just centering)
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_normalized = (X - X_mean) / X_std
    
    # Shuffle indices and create folds
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_folds
    
    # Store predictions for each sample
    all_predictions = np.zeros(n_samples)
    fold_assignments = np.zeros(n_samples, dtype=int)
    fold_losses = []
    
    print(f"\nRunning {n_folds}-fold cross-validation on {n_samples} samples...")
    print(f"Each fold: ~{fold_size} validation samples, ~{n_samples - fold_size} training samples")
    
    for fold in range(n_folds):
        # Define validation indices for this fold
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
        
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        # Get train/val data
        X_train, y_train = X_normalized[train_idx], y[train_idx]
        X_val, y_val = X_normalized[val_idx], y[val_idx]
        
        # Train and predict
        val_pred_log, train_loss = train_single_fold(
            X_train, y_train, X_val, y_val,
            hidden_dims, dropout, lr, weight_decay, n_epochs, batch_size, device
        )
        
        # Store predictions (convert back from log space)
        all_predictions[val_idx] = 10 ** val_pred_log
        fold_assignments[val_idx] = fold
        fold_losses.append(train_loss)
        
        # Compute validation metrics for this fold
        val_mae = np.mean(np.abs(10**val_pred_log - ages[val_idx]))
        val_corr = np.corrcoef(10**val_pred_log, ages[val_idx])[0, 1]
        
        print(f"  Fold {fold+1}/{n_folds}: train_loss={train_loss:.4f}, "
              f"val_MAE={val_mae:.1f} Myr, val_corr={val_corr:.3f}")
    
    return all_predictions, ages, tic_ids, fold_assignments, fold_losses


def plot_kfold_results(predictions, true_ages, bprp0, tic_ids, fold_assignments, output_dir):
    """Create comprehensive plots for k-fold results."""
    
    # Compute metrics
    mae = np.mean(np.abs(predictions - true_ages))
    rmse = np.sqrt(np.mean((predictions - true_ages) ** 2))
    log_mae = np.mean(np.abs(np.log10(predictions) - np.log10(true_ages)))
    corr = np.corrcoef(predictions, true_ages)[0, 1]
    
    # Median absolute error (more robust)
    med_ae = np.median(np.abs(predictions - true_ages))
    
    # Log-space metrics
    log_residuals = np.log10(predictions) - np.log10(true_ages)
    log_bias = np.mean(log_residuals)
    log_scatter = np.std(log_residuals)
    
    print(f"\n=== K-Fold Cross-Validation Results ===")
    print(f"MAE: {mae:.1f} Myr")
    print(f"Median AE: {med_ae:.1f} Myr")
    print(f"RMSE: {rmse:.1f} Myr")
    print(f"Log MAE: {log_mae:.3f} dex")
    print(f"Log bias: {log_bias:.3f} dex")
    print(f"Log scatter: {log_scatter:.3f} dex")
    print(f"Correlation: {corr:.3f}")
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Main scatter plot: predicted vs true age (log-log)
    ax1 = fig.add_subplot(2, 2, 1)
    scatter = ax1.scatter(true_ages, predictions, c=bprp0, cmap='coolwarm', 
                          alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    
    # Perfect prediction line
    min_age, max_age = true_ages.min(), true_ages.max()
    ax1.plot([min_age, max_age], [min_age, max_age], 'k--', lw=2, label='Perfect', zorder=10)
    
    # Factor of 2 lines
    ax1.plot([min_age, max_age], [min_age*2, max_age*2], 'k:', alpha=0.5, label='±2x')
    ax1.plot([min_age, max_age], [min_age/2, max_age/2], 'k:', alpha=0.5)
    
    ax1.set_xlabel('True Age (Myr)', fontsize=12)
    ax1.set_ylabel('Predicted Age (Myr)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title(f'K-Fold Age Predictions (r={corr:.3f})', fontsize=14)
    ax1.legend(loc='upper left')
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('BP-RP$_0$', fontsize=10)
    
    # Add metrics text box
    textstr = f'N = {len(true_ages)}\nMAE = {mae:.0f} Myr\nLog scatter = {log_scatter:.2f} dex\nr = {corr:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # 2. Residuals vs true age
    ax2 = fig.add_subplot(2, 2, 2)
    residuals = predictions - true_ages
    ax2.scatter(true_ages, residuals, c=bprp0, cmap='coolwarm', alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    ax2.axhline(0, color='k', linestyle='--', lw=2)
    ax2.set_xlabel('True Age (Myr)', fontsize=12)
    ax2.set_ylabel('Residual (Pred - True) [Myr]', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_title('Residuals vs True Age', fontsize=14)
    
    # 3. Log residuals histogram
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(log_residuals, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='r', linestyle='--', lw=2, label='Zero')
    ax3.axvline(log_bias, color='orange', linestyle='-', lw=2, label=f'Mean={log_bias:.2f}')
    ax3.set_xlabel('Log Residual (dex)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title(f'Log Residuals Distribution (σ={log_scatter:.2f} dex)', fontsize=14)
    ax3.legend()
    
    # 4. Residuals vs BPRP0 (check for color-dependent bias)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(bprp0, log_residuals, c=np.log10(true_ages), cmap='viridis', alpha=0.6, s=15)
    ax4.axhline(0, color='k', linestyle='--', lw=2)
    ax4.set_xlabel('BP-RP$_0$', fontsize=12)
    ax4.set_ylabel('Log Residual (dex)', fontsize=12)
    ax4.set_title('Residuals vs Color (colored by log age)', fontsize=14)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(np.log10(true_ages).min(), np.log10(true_ages).max()))
    cbar = plt.colorbar(sm, ax=ax4)
    cbar.set_label('log(Age/Myr)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kfold_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved k-fold results plot to {output_dir / 'kfold_results.png'}")
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'TIC_ID': tic_ids,
        'true_age_myr': true_ages,
        'predicted_age_myr': predictions,
        'bprp0': bprp0,
        'fold': fold_assignments,
        'log_residual': log_residuals,
    })
    csv_path = output_dir / 'kfold_predictions.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")
    
    # Save metrics
    metrics = {
        'n_samples': len(true_ages),
        'mae_myr': float(mae),
        'median_ae_myr': float(med_ae),
        'rmse_myr': float(rmse),
        'log_mae_dex': float(log_mae),
        'log_bias_dex': float(log_bias),
        'log_scatter_dex': float(log_scatter),
        'correlation': float(corr),
    }
    with open(output_dir / 'kfold_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='K-fold cross-validation for age prediction')
    
    # Data sources
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained BiDirectionalMinGRU model')
    parser.add_argument('--h5_path', type=str, default='data/timeseries.h5', help='Path to H5 data file')
    parser.add_argument('--pickle_path', type=str, default='data/star_sector_lc_formatted.pickle', help='Path to pickle file')
    parser.add_argument('--age_csv', type=str, default='data/TIC_cf_data.csv', help='Path to CSV with ages and BPRP0')
    parser.add_argument('--output_dir', type=str, default='output/age_predictor/kfold', help='Output directory')
    
    # Model extraction settings
    parser.add_argument('--hidden_size', type=int, default=64, help='RNN hidden size')
    parser.add_argument('--direction', type=str, default='bi', choices=['forward', 'backward', 'bi'])
    parser.add_argument('--mode', type=str, default='parallel', choices=['sequential', 'parallel'])
    parser.add_argument('--use_flow', action='store_true', help='Whether RNN uses flow head')
    parser.add_argument('--max_length', type=int, default=2048, help='Max sequence length')
    parser.add_argument('--pooling_mode', type=str, default='mean', choices=['mean', 'multiscale', 'final'])
    
    # Data selection
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to use')
    parser.add_argument('--stratify_by_age', action='store_true', help='Stratified sampling by age')
    parser.add_argument('--n_age_bins', type=int, default=20, help='Number of age bins for stratification')
    parser.add_argument('--bprp0_min', type=float, default=None, help='Min BPRP0 cut')
    parser.add_argument('--bprp0_max', type=float, default=None, help='Max BPRP0 cut')
    
    # K-fold settings
    parser.add_argument('--n_folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # MLP architecture
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32], help='MLP hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract latent vectors
    print("\n=== Extracting latent vectors ===")
    latent_vectors, ages, bprp0, tic_ids = load_data_fresh(
        model_path=args.model_path,
        h5_path=args.h5_path,
        pickle_path=args.pickle_path,
        csv_path=args.age_csv,
        hidden_size=args.hidden_size,
        direction=args.direction,
        mode=args.mode,
        use_flow=args.use_flow,
        max_length=args.max_length,
        batch_size=32,
        pooling_mode=args.pooling_mode,
        max_samples=args.max_samples,
        stratify_by_age=args.stratify_by_age,
        n_age_bins=args.n_age_bins,
        bprp0_min=args.bprp0_min,
        bprp0_max=args.bprp0_max,
        device=device,
    )
    
    # Run k-fold cross-validation
    print("\n=== Running K-Fold Cross-Validation ===")
    predictions, true_ages, tic_ids, fold_assignments, fold_losses = run_kfold_cv(
        latent_vectors=latent_vectors,
        ages=ages,
        bprp0=bprp0,
        tic_ids=tic_ids,
        n_folds=args.n_folds,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    
    # Plot and save results
    metrics = plot_kfold_results(predictions, true_ages, bprp0, tic_ids, fold_assignments, output_dir)
    
    # Save config
    config = vars(args)
    config['latent_dim'] = latent_vectors.shape[1]
    config['n_samples'] = len(ages)
    config['fold_losses'] = [float(l) for l in fold_losses]
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
