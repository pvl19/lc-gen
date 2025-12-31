"""K-fold cross-validation for age prediction using only BPRP0 + Prot (gyrochronology baseline).

This script serves as a baseline comparison for the latent-space age predictor.
It uses only the rotation period (Prot) and color (BPRP0) to predict stellar age,
which is the traditional gyrochronology approach.

Usage:
    python scripts/kfold_gyro_baseline.py \
        --age_csv data/TIC_cf_data.csv \
        --output_dir output/age_predictor/gyro_baseline \
        --n_folds 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class GyroAgePredictor(nn.Module):
    """MLP to predict log(age) from BPRP0 + Prot only."""
    
    def __init__(self, input_dim: int = 2, hidden_dims: list = [128, 64, 32], dropout: float = 0.2):
        """
        Args:
            input_dim: dimension of input (default 2: BPRP0 + Prot)
            hidden_dims: list of hidden layer dimensions
            dropout: dropout probability
        """
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
        
        # Output layer: predict log(age) as a single scalar
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """
        Args:
            x: (B, 2) tensor of [bprp0, log_prot]
        Returns:
            (B, 1) predicted log(age in Myr)
        """
        return self.mlp(x)


def load_gyro_data(csv_path: str, bprp0_min: float = None, bprp0_max: float = None):
    """Load BPRP0, Prot, and ages from CSV.
    
    Args:
        csv_path: path to CSV with stellar properties
        bprp0_min: minimum BPRP0 cut
        bprp0_max: maximum BPRP0 cut
    
    Returns:
        bprp0: (N,) array
        prot: (N,) array of rotation periods in days
        ages: (N,) array in Myr
        tic_ids: (N,) array of TIC IDs
    """
    df = pd.read_csv(csv_path)
    
    # Filter to valid data
    valid_mask = (~df['age_Myr'].isna() & 
                  ~df['BPRP0'].isna() & 
                  ~df['Prot'].isna() &
                  (df['Prot'] > 0))  # Prot must be positive for log
    
    if bprp0_min is not None:
        valid_mask &= (df['BPRP0'] >= bprp0_min)
    if bprp0_max is not None:
        valid_mask &= (df['BPRP0'] <= bprp0_max)
    
    df_valid = df[valid_mask]
    
    bprp0 = df_valid['BPRP0'].values
    prot = df_valid['Prot'].values
    ages = df_valid['age_Myr'].values
    tic_ids = df_valid['TIC_ID'].values
    
    print(f"Loaded {len(ages)} samples with valid BPRP0, Prot, and age")
    print(f"Age range: {ages.min():.1f} - {ages.max():.1f} Myr")
    print(f"BPRP0 range: {bprp0.min():.2f} - {bprp0.max():.2f}")
    print(f"Prot range: {prot.min():.2f} - {prot.max():.2f} days")
    
    return bprp0, prot, ages, tic_ids


def train_single_fold(X_train, y_train, X_val, y_val, 
                      hidden_dims, dropout, lr, weight_decay, n_epochs, batch_size, device):
    """Train a single model on training data, return predictions on validation data.
    
    Returns:
        val_pred: (N,) validation predictions (log age)
        best_train_loss: best training loss
        best_state: best model state dict (on CPU)
    """
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=device),
        torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = GyroAgePredictor(X_train.shape[1], hidden_dims, dropout).to(device)
    
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
    
    return val_pred, best_train_loss, best_state


def run_kfold_cv(bprp0, prot, ages, tic_ids,
                 n_folds=10, hidden_dims=[128, 64, 32], dropout=0.2,
                 lr=1e-3, weight_decay=1e-4, n_epochs=100, batch_size=64,
                 device=None, seed=42, save_models_dir=None):
    """Run k-fold cross-validation.
    
    Args:
        save_models_dir: if provided, save fold models to this directory
    
    Returns:
        predictions: array of predicted ages for all samples
        true_ages: array of true ages
        bprp0: array of BPRP0 values
        prot: array of rotation periods
        tic_ids: array of TIC IDs
        fold_indices: which fold each sample was in
        fold_losses: list of training losses per fold
        normalization_params: dict with X_mean, X_std
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(seed)
    n_samples = len(ages)
    
    # Prepare features: use log(Prot) for better scaling
    log_prot = np.log10(prot)
    X = np.column_stack([bprp0, log_prot])
    y = np.log10(ages)  # Predict in log space
    
    # Normalize inputs
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_normalized = (X - X_mean) / X_std
    
    # Store normalization params for inference
    normalization_params = {
        'X_mean': X_mean,
        'X_std': X_std,
        'input_dim': X.shape[1],
        'feature_names': ['BPRP0', 'log10_Prot'],
    }
    
    # Shuffle indices and create folds
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_folds
    
    # Store predictions for each sample
    all_predictions = np.zeros(n_samples)
    fold_assignments = np.zeros(n_samples, dtype=int)
    fold_losses = []
    fold_models = []
    
    print(f"\nRunning {n_folds}-fold cross-validation on {n_samples} samples...")
    print(f"Each fold: ~{fold_size} validation samples, ~{n_samples - fold_size} training samples")
    print(f"Input features: BPRP0, log10(Prot)")
    
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
        val_pred_log, train_loss, model_state = train_single_fold(
            X_train, y_train, X_val, y_val,
            hidden_dims, dropout, lr, weight_decay, n_epochs, batch_size, device
        )
        
        fold_models.append(model_state)
        
        # Store predictions (convert back from log space)
        all_predictions[val_idx] = 10 ** val_pred_log
        fold_assignments[val_idx] = fold
        fold_losses.append(train_loss)
        
        # Compute validation metrics for this fold
        val_mae = np.mean(np.abs(10**val_pred_log - ages[val_idx]))
        val_corr = np.corrcoef(10**val_pred_log, ages[val_idx])[0, 1]
        
        print(f"  Fold {fold+1}/{n_folds}: train_loss={train_loss:.4f}, "
              f"val_MAE={val_mae:.1f} Myr, val_corr={val_corr:.3f}")
    
    # Save models if directory provided
    if save_models_dir is not None:
        save_dir = Path(save_models_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all fold models and normalization params
        save_dict = {
            'fold_models': fold_models,
            'normalization_params': normalization_params,
            'hidden_dims': hidden_dims,
            'dropout': dropout,
            'n_folds': n_folds,
            'seed': seed,
            'fold_indices': indices,
        }
        model_path = save_dir / 'gyro_kfold_models.pt'
        torch.save(save_dict, model_path)
        print(f"\nSaved {n_folds} fold models to {model_path}")
    
    return all_predictions, ages, bprp0, prot, tic_ids, fold_assignments, fold_losses, normalization_params


def plot_results(predictions, true_ages, bprp0, prot, tic_ids, fold_assignments, output_dir):
    """Create comprehensive plots for k-fold results."""
    
    # Compute metrics
    mae = np.mean(np.abs(predictions - true_ages))
    rmse = np.sqrt(np.mean((predictions - true_ages) ** 2))
    log_mae = np.mean(np.abs(np.log10(predictions) - np.log10(true_ages)))
    corr = np.corrcoef(predictions, true_ages)[0, 1]
    med_ae = np.median(np.abs(predictions - true_ages))
    
    # Log-space metrics
    log_residuals = np.log10(predictions) - np.log10(true_ages)
    log_bias = np.mean(log_residuals)
    log_scatter = np.std(log_residuals)
    
    print(f"\n=== Gyro Baseline K-Fold Results ===")
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
    
    min_age, max_age = true_ages.min(), true_ages.max()
    ax1.plot([min_age, max_age], [min_age, max_age], 'k--', lw=2, label='Perfect', zorder=10)
    ax1.plot([min_age, max_age], [min_age*2, max_age*2], 'k:', alpha=0.5, label='±2x')
    ax1.plot([min_age, max_age], [min_age/2, max_age/2], 'k:', alpha=0.5)
    
    ax1.set_xlabel('True Age (Myr)', fontsize=12)
    ax1.set_ylabel('Predicted Age (Myr)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title(f'Gyro Baseline: BPRP0 + Prot Only (r={corr:.3f})', fontsize=14)
    ax1.legend(loc='upper left')
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('BP-RP$_0$', fontsize=10)
    
    textstr = f'N = {len(true_ages)}\nMAE = {mae:.0f} Myr\nLog scatter = {log_scatter:.2f} dex\nr = {corr:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # 2. Residuals vs true age
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(true_ages, log_residuals, c=bprp0, cmap='coolwarm', alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    ax2.axhline(0, color='k', linestyle='--', lw=2)
    ax2.set_xlabel('True Age (Myr)', fontsize=12)
    ax2.set_ylabel('Log Residual (dex)', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_title('Log Residuals vs True Age', fontsize=14)
    
    # 3. Log residuals histogram
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(log_residuals, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='r', linestyle='--', lw=2, label='Zero')
    ax3.axvline(log_bias, color='orange', linestyle='-', lw=2, label=f'Mean={log_bias:.2f}')
    ax3.set_xlabel('Log Residual (dex)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title(f'Log Residuals Distribution (σ={log_scatter:.2f} dex)', fontsize=14)
    ax3.legend()
    
    # 4. Residuals vs Prot
    ax4 = fig.add_subplot(2, 2, 4)
    scatter4 = ax4.scatter(prot, log_residuals, c=np.log10(true_ages), cmap='viridis', alpha=0.6, s=15)
    ax4.axhline(0, color='k', linestyle='--', lw=2)
    ax4.set_xlabel('Rotation Period (days)', fontsize=12)
    ax4.set_ylabel('Log Residual (dex)', fontsize=12)
    ax4.set_xscale('log')
    ax4.set_title('Residuals vs Prot (colored by log age)', fontsize=14)
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('log(Age/Myr)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gyro_kfold_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved gyro baseline plot to {output_dir / 'gyro_kfold_results.png'}")
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'TIC_ID': tic_ids,
        'true_age_myr': true_ages,
        'predicted_age_myr': predictions,
        'bprp0': bprp0,
        'prot_days': prot,
        'fold': fold_assignments,
        'log_residual': log_residuals,
    })
    csv_path = output_dir / 'gyro_kfold_predictions.csv'
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
    with open(output_dir / 'gyro_kfold_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='K-fold gyrochronology baseline (BPRP0 + Prot only)')
    
    parser.add_argument('--age_csv', type=str, default='data/TIC_cf_data.csv', help='Path to CSV with ages, BPRP0, Prot')
    parser.add_argument('--output_dir', type=str, default='output/age_predictor/gyro_baseline', help='Output directory')
    
    # Data selection
    parser.add_argument('--bprp0_min', type=float, default=None, help='Min BPRP0 cut')
    parser.add_argument('--bprp0_max', type=float, default=None, help='Max BPRP0 cut')
    
    # MLP architecture (same as latent-space model for fair comparison)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32], help='MLP hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n=== Loading gyro data ===")
    bprp0, prot, ages, tic_ids = load_gyro_data(
        args.age_csv,
        bprp0_min=args.bprp0_min,
        bprp0_max=args.bprp0_max
    )
    
    # Run k-fold CV
    print("\n=== Running k-fold cross-validation ===")
    predictions, true_ages, bprp0, prot, tic_ids, fold_assignments, fold_losses, norm_params = run_kfold_cv(
        bprp0, prot, ages, tic_ids,
        n_folds=args.n_folds,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        save_models_dir=output_dir,  # Save models to output directory
    )
    
    # Plot and save results
    metrics = plot_results(predictions, true_ages, bprp0, prot, tic_ids, fold_assignments, output_dir)
    
    # Save config
    config = vars(args)
    config['n_samples'] = len(ages)
    config['input_features'] = ['BPRP0', 'log10(Prot)']
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
