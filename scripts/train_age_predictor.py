"""Train an MLP to predict stellar age from latent space representations + BPRP0.

This is a downstream module that uses pre-extracted latent vectors from the
BiDirectionalMinGRU model to predict stellar ages. It does not modify or
require the original RNN model during training.

Usage:
    python scripts/train_age_predictor.py \
        --latent_npz output/simple_rnn/umap_plots/umap_age_baseline_v15_masking_real_60e_mean_data.npz \
        --age_csv data/TIC_cf_data.csv \
        --pickle_path data/star_sector_lc_formatted.pickle \
        --output_dir output/age_predictor
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

# Ensure local 'src' is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class AgePredictor(nn.Module):
    """Simple MLP to predict log(age) from latent vectors + BPRP0."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], dropout: float = 0.2):
        """
        Args:
            input_dim: dimension of input (latent_dim + 1 for BPRP0)
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
            x: (B, input_dim) tensor of [latent_vector, bprp0]
        Returns:
            (B, 1) predicted log(age in Myr)
        """
        return self.mlp(x)


def load_data_from_npz(npz_path: str, csv_path: str, pickle_path: str, sample_indices: np.ndarray = None):
    """Load latent vectors, ages, and BPRP0 values.
    
    Args:
        npz_path: path to .npz file with latent vectors (from UMAP script)
        csv_path: path to CSV with stellar properties
        pickle_path: path to pickle with TIC IDs
        sample_indices: indices used when creating the npz file (for TIC ID mapping)
    
    Returns:
        latent_vectors: (N, D) array
        ages: (N,) array in Myr
        bprp0: (N,) array
    """
    # Load pre-extracted latent vectors
    data = np.load(npz_path)
    latent_vectors = data['latent_vectors']
    ages = data['ages']
    
    # We need to get BPRP0 values - load from CSV using TIC IDs
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)
    
    metadatas = pickle_data['metadatas']
    if sample_indices is not None:
        metadatas = [metadatas[i] for i in sample_indices]
    
    # Load CSV for BPRP0
    age_df = pd.read_csv(csv_path)
    tic_to_bprp0 = dict(zip(age_df['TIC_ID'], age_df['BPRP0']))
    
    # The npz file only contains samples with valid ages, so we need to match
    # We'll load BPRP0 for all samples that were used in the npz
    # Since ages in npz are already filtered, we need the TIC IDs that correspond
    
    # Actually, the npz saves ages_valid which is already filtered
    # We need to reconstruct which samples those were
    
    print(f"Loaded {len(latent_vectors)} latent vectors with shape {latent_vectors.shape}")
    print(f"Ages range: {ages.min():.1f} - {ages.max():.1f} Myr")
    
    return latent_vectors, ages


def load_data_fresh(model_path: str, h5_path: str, pickle_path: str, csv_path: str,
                    hidden_size: int, direction: str, mode: str, use_flow: bool,
                    max_length: int, batch_size: int, pooling_mode: str,
                    max_samples: int = None, stratify_by_age: bool = False,
                    n_age_bins: int = 20, bprp0_min: float = None, bprp0_max: float = None,
                    device: torch.device = None):
    """Extract latent vectors fresh from the model (reusing UMAP script logic).
    
    This allows training with different pooling modes without re-running UMAP.
    """
    # Import the extraction functions from the UMAP script
    from plot_umap_latent import (
        load_model, extract_latent_vectors, load_ages, get_stratified_indices
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine which samples to use (same logic as UMAP script)
    sample_indices = None
    all_ages, all_bprp0, _ = load_ages(pickle_path, csv_path, sample_indices=None)
    
    # Apply BPRP0 cuts
    valid_mask = ~np.isnan(all_ages)
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
    
    # Filter to valid ages (should already be mostly valid due to stratification)
    valid_mask = ~np.isnan(ages) & ~np.isnan(bprp0)
    latent_vectors = latent_vectors[valid_mask]
    ages = ages[valid_mask]
    bprp0 = bprp0[valid_mask]
    
    print(f"Extracted {len(latent_vectors)} samples with valid ages and BPRP0")
    print(f"Latent dimension: {latent_vectors.shape[1]}")
    print(f"Age range: {ages.min():.1f} - {ages.max():.1f} Myr")
    print(f"BPRP0 range: {bprp0.min():.2f} - {bprp0.max():.2f}")
    
    return latent_vectors, ages, bprp0


def train_age_predictor(
    latent_vectors: np.ndarray,
    ages: np.ndarray,
    bprp0: np.ndarray,
    hidden_dims: list = [128, 64, 32],
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 100,
    batch_size: int = 64,
    val_split: float = 0.2,
    device: torch.device = None,
    verbose: bool = True,
):
    """Train the age predictor MLP.
    
    Args:
        latent_vectors: (N, D) latent representations
        ages: (N,) ages in Myr
        bprp0: (N,) BPRP0 values
        hidden_dims: MLP hidden layer dimensions
        dropout: dropout probability
        lr: learning rate
        weight_decay: L2 regularization
        n_epochs: number of training epochs
        batch_size: batch size
        val_split: fraction for validation
        device: torch device
        verbose: print progress
    
    Returns:
        model: trained AgePredictor
        history: dict with training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data: concatenate latent vectors with BPRP0
    # Normalize BPRP0 to similar scale as latent vectors
    bprp0_normalized = (bprp0 - bprp0.mean()) / (bprp0.std() + 1e-8)
    
    X = np.concatenate([latent_vectors, bprp0_normalized[:, None]], axis=1)
    
    # Predict log(age) for better scaling (ages span orders of magnitude)
    y = np.log10(ages)
    
    # Normalize inputs (latent vectors)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    # Train/val split
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * val_split)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(-1)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    input_dim = X.shape[1]
    model = AgePredictor(input_dim, hidden_dims, dropout).to(device)
    
    if verbose:
        print(f"Model parameters: {model.n_params:,}")
        print(f"Input dimension: {input_dim} (latent: {latent_vectors.shape[1]}, BPRP0: 1)")
        print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(train_indices)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            
            # MAE in original age units (Myr)
            val_pred_age = 10 ** val_pred.cpu().numpy()
            val_true_age = 10 ** y_val.cpu().numpy()
            val_mae = np.mean(np.abs(val_pred_age - val_true_age))
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_MAE={val_mae:.1f} Myr")
    
    # Restore best model
    model.load_state_dict(best_state)
    
    # Store normalization params for inference
    model.X_mean = X_mean
    model.X_std = X_std
    model.bprp0_mean = bprp0.mean()
    model.bprp0_std = bprp0.std()
    
    return model, history


def evaluate_model(model, latent_vectors, ages, bprp0, device=None):
    """Evaluate the trained model and return metrics."""
    if device is None:
        device = next(model.parameters()).device
    
    # Prepare data same way as training
    bprp0_normalized = (bprp0 - model.bprp0_mean) / (model.bprp0_std + 1e-8)
    X = np.concatenate([latent_vectors, bprp0_normalized[:, None]], axis=1)
    X = (X - model.X_mean) / model.X_std
    
    X = torch.tensor(X, dtype=torch.float32, device=device)
    
    model.eval()
    with torch.no_grad():
        log_age_pred = model(X).cpu().numpy().squeeze()
    
    age_pred = 10 ** log_age_pred
    
    # Metrics
    mae = np.mean(np.abs(age_pred - ages))
    rmse = np.sqrt(np.mean((age_pred - ages) ** 2))
    
    # Log-space metrics (more meaningful for ages spanning orders of magnitude)
    log_mae = np.mean(np.abs(log_age_pred - np.log10(ages)))
    
    # Correlation
    corr = np.corrcoef(age_pred, ages)[0, 1]
    
    return {
        'mae_myr': mae,
        'rmse_myr': rmse,
        'log_mae': log_mae,
        'correlation': corr,
        'predictions': age_pred,
        'true_ages': ages,
    }


def plot_results(history, eval_results, output_dir: Path):
    """Plot training curves and prediction scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training curves
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (log age)')
    ax.set_title('Training Loss')
    ax.legend()
    ax.set_yscale('log')
    
    # Validation MAE
    ax = axes[1]
    ax.plot(history['val_mae'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (Myr)')
    ax.set_title('Validation MAE')
    
    # Prediction scatter
    ax = axes[2]
    ax.scatter(eval_results['true_ages'], eval_results['predictions'], alpha=0.5, s=10)
    
    # Perfect prediction line
    min_age = min(eval_results['true_ages'].min(), eval_results['predictions'].min())
    max_age = max(eval_results['true_ages'].max(), eval_results['predictions'].max())
    ax.plot([min_age, max_age], [min_age, max_age], 'r--', label='Perfect')
    
    ax.set_xlabel('True Age (Myr)')
    ax.set_ylabel('Predicted Age (Myr)')
    ax.set_title(f'Age Predictions (r={eval_results["correlation"]:.3f})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training results plot to {output_dir / 'training_results.png'}")


def main():
    parser = argparse.ArgumentParser(description='Train MLP to predict stellar age from latent space + BPRP0')
    
    # Data sources
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained BiDirectionalMinGRU model')
    parser.add_argument('--h5_path', type=str, default='data/timeseries.h5', help='Path to H5 data file')
    parser.add_argument('--pickle_path', type=str, default='data/star_sector_lc_formatted.pickle', help='Path to pickle file')
    parser.add_argument('--age_csv', type=str, default='data/TIC_cf_data.csv', help='Path to CSV with ages and BPRP0')
    parser.add_argument('--output_dir', type=str, default='output/age_predictor', help='Output directory')
    
    # Model extraction settings (same as UMAP script)
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
    
    # MLP architecture
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32], help='MLP hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split fraction')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract latent vectors
    print("\n=== Extracting latent vectors ===")
    latent_vectors, ages, bprp0 = load_data_fresh(
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
    
    # Train model
    print("\n=== Training age predictor ===")
    model, history = train_age_predictor(
        latent_vectors=latent_vectors,
        ages=ages,
        bprp0=bprp0,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        device=device,
    )
    
    # Evaluate
    print("\n=== Evaluation ===")
    eval_results = evaluate_model(model, latent_vectors, ages, bprp0, device)
    print(f"MAE: {eval_results['mae_myr']:.1f} Myr")
    print(f"RMSE: {eval_results['rmse_myr']:.1f} Myr")
    print(f"Log MAE: {eval_results['log_mae']:.3f} dex")
    print(f"Correlation: {eval_results['correlation']:.3f}")
    
    # Plot results
    plot_results(history, eval_results, output_dir)
    
    # Save model and config
    save_dict = {
        'model_state_dict': model.state_dict(),
        'X_mean': model.X_mean,
        'X_std': model.X_std,
        'bprp0_mean': model.bprp0_mean,
        'bprp0_std': model.bprp0_std,
        'input_dim': latent_vectors.shape[1] + 1,
        'hidden_dims': args.hidden_dims,
        'dropout': args.dropout,
        'pooling_mode': args.pooling_mode,
        'metrics': {
            'mae_myr': eval_results['mae_myr'],
            'rmse_myr': eval_results['rmse_myr'],
            'log_mae': eval_results['log_mae'],
            'correlation': eval_results['correlation'],
        }
    }
    
    model_save_path = output_dir / 'age_predictor.pt'
    torch.save(save_dict, model_save_path)
    print(f"\nSaved model to {model_save_path}")
    
    # Save config
    config = vars(args)
    config['latent_dim'] = latent_vectors.shape[1]
    config['n_samples'] = len(ages)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Done!")


if __name__ == '__main__':
    main()
