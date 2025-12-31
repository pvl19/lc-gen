#!/usr/bin/env python
"""
Standalone inference script for predicting stellar ages using saved k-fold models.

Loads pre-trained MLP models from kfold_age_inference.py or kfold_gyro_baseline.py
and predicts ages on new data without retraining.

Usage:
    # Predict using latent space model
    python scripts/predict_ages.py \
        --model_file output/age_predictor/kfold/kfold_models.pt \
        --rnn_model output/simple_rnn/models/baseline_v15_masking_real_60e.pt \
        --h5_path data/timeseries.h5 \
        --pickle_path data/star_sector_lc_formatted.pickle \
        --age_csv data/TIC_cf_data.csv \
        --output_csv output/age_predictor/predictions_v1.csv
    
    # Predict using gyro baseline model
    python scripts/predict_ages.py \
        --model_file output/age_predictor/gyro_baseline/gyro_kfold_models.pt \
        --age_csv data/TIC_cf_data.csv \
        --output_csv output/age_predictor/gyro_predictions_v1.csv \
        --model_type gyro
"""

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


# =============================================================================
# Model Definitions (must match training)
# =============================================================================

class AgePredictor(nn.Module):
    """MLP for age prediction from latent space + BPRP0."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class GyroAgePredictor(nn.Module):
    """MLP for age prediction from BPRP0 + log(Prot)."""
    
    def __init__(self, input_dim: int = 2, hidden_dims: list = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


# =============================================================================
# Latent Space Extraction (for latent model only)
# =============================================================================

def load_rnn_model(model_path, hidden_size, direction, mode, use_flow, device):
    """Load pre-trained BiDirectionalMinGRU model."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    
    from lcgen.models.simple_rnn import BiDirectionalMinGRU
    
    model = BiDirectionalMinGRU(
        input_size=3,  # flux, flux_err, dt
        hidden_size=hidden_size,
        mode=mode,
        direction=direction,
        use_flow_head=use_flow,
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def extract_latent_vectors(rnn_model, h5_path, pickle_path, max_length, 
                           pooling_mode, batch_size, device, tic_ids_needed=None):
    """Extract latent vectors for specified TIC IDs.
    
    Args:
        rnn_model: loaded BiDirectionalMinGRU
        h5_path: path to H5 file with time series
        pickle_path: path to pickle with TIC ID mapping
        max_length: max sequence length
        pooling_mode: 'mean', 'final', or 'multiscale'
        batch_size: batch size for extraction
        device: torch device
        tic_ids_needed: array of TIC IDs to extract (if None, extract all)
    
    Returns:
        latent_vectors: (N, D) array
        tic_ids: (N,) array of TIC IDs
    """
    # Load pickle to get TIC ID mapping
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)
    
    # Build index -> TIC ID mapping
    if isinstance(pickle_data, dict) and 'tic_id' in pickle_data:
        all_tic_ids = np.array(pickle_data['tic_id'])
    else:
        all_tic_ids = np.array([item['tic_id'] for item in pickle_data])
    
    # Load H5 file
    h5f = h5py.File(h5_path, 'r')
    flux = h5f['flux'][:]
    flux_err = h5f['flux_err'][:]
    time = h5f['time'][:]
    length = h5f['length'][:]
    
    # Determine which indices to extract
    if tic_ids_needed is not None:
        tic_set = set(tic_ids_needed)
        indices = [i for i, tid in enumerate(all_tic_ids) if tid in tic_set]
    else:
        indices = list(range(len(all_tic_ids)))
    
    latent_list = []
    tic_list = []
    
    for batch_start in tqdm(range(0, len(indices), batch_size), desc="Extracting latent vectors"):
        batch_indices = indices[batch_start:batch_start + batch_size]
        
        batch_flux = []
        batch_flux_err = []
        batch_time = []
        batch_lengths = []
        
        for idx in batch_indices:
            L = min(int(length[idx]), max_length)
            batch_flux.append(flux[idx, :L])
            batch_flux_err.append(flux_err[idx, :L])
            batch_time.append(time[idx, :L])
            batch_lengths.append(L)
        
        # Pad to max length in batch
        max_len = max(batch_lengths)
        B = len(batch_indices)
        
        flux_padded = np.zeros((B, max_len), dtype=np.float32)
        flux_err_padded = np.zeros((B, max_len), dtype=np.float32)
        time_padded = np.zeros((B, max_len), dtype=np.float32)
        mask = np.zeros((B, max_len), dtype=np.float32)
        
        for i, L in enumerate(batch_lengths):
            flux_padded[i, :L] = batch_flux[i]
            flux_err_padded[i, :L] = batch_flux_err[i]
            time_padded[i, :L] = batch_time[i]
            mask[i, :L] = 1.0
        
        # Compute dt
        dt_padded = np.zeros_like(time_padded)
        dt_padded[:, 1:] = np.diff(time_padded, axis=1)
        
        # Create input tensor
        x = np.stack([flux_padded, flux_err_padded, dt_padded], axis=-1)
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
        
        # Forward pass
        with torch.no_grad():
            if hasattr(rnn_model, 'use_flow_head') and rnn_model.use_flow_head:
                hidden, _ = rnn_model(x_t, mask_t)
            else:
                hidden = rnn_model(x_t, mask_t)
        
        # Apply pooling
        if pooling_mode == 'mean':
            # Mean pooling over valid timesteps
            mask_exp = mask_t.unsqueeze(-1)
            latent = (hidden * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-8)
        elif pooling_mode == 'final':
            # Get final hidden state for each sequence
            batch_latent = []
            for i, L in enumerate(batch_lengths):
                batch_latent.append(hidden[i, L-1, :])
            latent = torch.stack(batch_latent, dim=0)
        elif pooling_mode == 'multiscale':
            latent = compute_multiscale_features(hidden, mask_t)
        else:
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}")
        
        latent_list.append(latent.cpu().numpy())
        tic_list.extend([all_tic_ids[idx] for idx in batch_indices])
    
    h5f.close()
    
    latent_vectors = np.concatenate(latent_list, axis=0)
    tic_ids = np.array(tic_list)
    
    return latent_vectors, tic_ids


def compute_multiscale_features(hidden, mask):
    """Compute multi-scale features from hidden states."""
    B, T, D = hidden.shape
    mask_exp = mask.unsqueeze(-1)
    valid_counts = mask_exp.sum(dim=1, keepdim=True) + 1e-8
    
    features = []
    
    # Global mean
    global_mean = (hidden * mask_exp).sum(dim=1) / valid_counts.squeeze(1)
    features.append(global_mean)
    
    # Global max
    hidden_masked = hidden.clone()
    hidden_masked[mask == 0] = -1e9
    global_max, _ = hidden_masked.max(dim=1)
    features.append(global_max)
    
    # Global std
    mean_exp = global_mean.unsqueeze(1)
    sq_diff = ((hidden - mean_exp) ** 2) * mask_exp
    global_std = torch.sqrt(sq_diff.sum(dim=1) / valid_counts.squeeze(1) + 1e-8)
    features.append(global_std)
    
    # First/last quartile means
    for start_frac, end_frac in [(0.0, 0.25), (0.75, 1.0)]:
        chunk_feats = []
        for i in range(B):
            L = int(mask[i].sum().item())
            if L > 0:
                s = int(start_frac * L)
                e = max(s + 1, int(end_frac * L))
                chunk_feats.append(hidden[i, s:e, :].mean(dim=0))
            else:
                chunk_feats.append(torch.zeros(D, device=hidden.device))
        features.append(torch.stack(chunk_feats, dim=0))
    
    # Temporal chunks (4 chunks)
    for n_chunks in [4]:
        chunk_means = []
        for c in range(n_chunks):
            chunk_feats = []
            for i in range(B):
                L = int(mask[i].sum().item())
                if L > 0:
                    s = int(c * L / n_chunks)
                    e = int((c + 1) * L / n_chunks)
                    if e > s:
                        chunk_feats.append(hidden[i, s:e, :].mean(dim=0))
                    else:
                        chunk_feats.append(hidden[i, s, :])
                else:
                    chunk_feats.append(torch.zeros(D, device=hidden.device))
            chunk_means.append(torch.stack(chunk_feats, dim=0))
        features.extend(chunk_means)
    
    return torch.cat(features, dim=-1)


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_with_latent_model(checkpoint, latent_vectors, bprp0, device, ensemble='mean'):
    """Predict ages using loaded latent space k-fold models.
    
    Args:
        checkpoint: loaded checkpoint dict with fold_models and normalization_params
        latent_vectors: (N, D) latent representations
        bprp0: (N,) BPRP0 values
        device: torch device
        ensemble: 'mean' to average predictions, 'median' for median
    
    Returns:
        predictions: (N,) predicted ages in Myr
        predictions_std: (N,) std of predictions across folds
    """
    fold_models = checkpoint['fold_models']
    norm_params = checkpoint['normalization_params']
    hidden_dims = checkpoint['hidden_dims']
    dropout = checkpoint.get('dropout', 0.2)
    
    # Prepare features
    bprp0_mean = norm_params['bprp0_mean']
    bprp0_std = norm_params['bprp0_std']
    bprp0_normalized = (bprp0 - bprp0_mean) / (bprp0_std + 1e-8)
    
    X = np.concatenate([latent_vectors, bprp0_normalized[:, None]], axis=1)
    X_normalized = (X - norm_params['X_mean']) / norm_params['X_std']
    
    X_t = torch.tensor(X_normalized, dtype=torch.float32, device=device)
    
    # Create model
    input_dim = norm_params['input_dim']
    model = AgePredictor(input_dim, hidden_dims, dropout).to(device)
    
    # Predict with each fold
    fold_predictions = []
    for fold_state in fold_models:
        model.load_state_dict({k: v.to(device) for k, v in fold_state.items()})
        model.eval()
        
        with torch.no_grad():
            pred_log = model(X_t).cpu().numpy().squeeze()
        fold_predictions.append(pred_log)
    
    fold_predictions = np.array(fold_predictions)  # (n_folds, N)
    
    # Ensemble
    if ensemble == 'mean':
        ensemble_log = fold_predictions.mean(axis=0)
    else:
        ensemble_log = np.median(fold_predictions, axis=0)
    
    predictions = 10 ** ensemble_log
    predictions_std = (10 ** fold_predictions).std(axis=0)
    
    return predictions, predictions_std


def predict_with_gyro_model(checkpoint, bprp0, prot, device, ensemble='mean'):
    """Predict ages using loaded gyro k-fold models.
    
    Args:
        checkpoint: loaded checkpoint dict
        bprp0: (N,) BPRP0 values
        prot: (N,) rotation periods in days
        device: torch device
        ensemble: 'mean' or 'median'
    
    Returns:
        predictions: (N,) predicted ages in Myr
        predictions_std: (N,) std across folds
    """
    fold_models = checkpoint['fold_models']
    norm_params = checkpoint['normalization_params']
    hidden_dims = checkpoint['hidden_dims']
    dropout = checkpoint.get('dropout', 0.2)
    
    # Prepare features
    log_prot = np.log10(prot)
    X = np.column_stack([bprp0, log_prot])
    X_normalized = (X - norm_params['X_mean']) / norm_params['X_std']
    
    X_t = torch.tensor(X_normalized, dtype=torch.float32, device=device)
    
    # Create model
    input_dim = norm_params['input_dim']
    model = GyroAgePredictor(input_dim, hidden_dims, dropout).to(device)
    
    # Predict with each fold
    fold_predictions = []
    for fold_state in fold_models:
        model.load_state_dict({k: v.to(device) for k, v in fold_state.items()})
        model.eval()
        
        with torch.no_grad():
            pred_log = model(X_t).cpu().numpy().squeeze()
        fold_predictions.append(pred_log)
    
    fold_predictions = np.array(fold_predictions)
    
    if ensemble == 'mean':
        ensemble_log = fold_predictions.mean(axis=0)
    else:
        ensemble_log = np.median(fold_predictions, axis=0)
    
    predictions = 10 ** ensemble_log
    predictions_std = (10 ** fold_predictions).std(axis=0)
    
    return predictions, predictions_std


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Predict stellar ages using saved k-fold models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Latent space model prediction
  python scripts/predict_ages.py \\
      --model_file output/age_predictor/kfold/kfold_models.pt \\
      --rnn_model output/simple_rnn/models/baseline_v15_masking_real_60e.pt \\
      --age_csv data/TIC_cf_data.csv \\
      --output_csv predictions.csv

  # Gyro baseline prediction  
  python scripts/predict_ages.py \\
      --model_file output/age_predictor/gyro_baseline/gyro_kfold_models.pt \\
      --age_csv data/TIC_cf_data.csv \\
      --output_csv gyro_predictions.csv \\
      --model_type gyro
        """
    )
    
    # Required arguments
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to saved k-fold models (.pt file)')
    parser.add_argument('--age_csv', type=str, required=True,
                        help='Path to CSV with TIC_ID, BPRP0, Prot, age_Myr')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Output CSV path for predictions')
    
    # Model type
    parser.add_argument('--model_type', type=str, default='latent',
                        choices=['latent', 'gyro'],
                        help='Type of model: latent (RNN latent space) or gyro (BPRP0+Prot)')
    
    # For latent model only
    parser.add_argument('--rnn_model', type=str, default=None,
                        help='Path to pre-trained RNN model (required for latent type)')
    parser.add_argument('--h5_path', type=str, default='data/timeseries.h5',
                        help='Path to H5 file with time series')
    parser.add_argument('--pickle_path', type=str, default='data/star_sector_lc_formatted.pickle',
                        help='Path to pickle file with TIC ID mapping')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='RNN hidden size')
    parser.add_argument('--direction', type=str, default='bi',
                        choices=['forward', 'backward', 'bi'])
    parser.add_argument('--mode', type=str, default='parallel',
                        choices=['sequential', 'parallel'])
    parser.add_argument('--use_flow', action='store_true',
                        help='Whether RNN uses flow head')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Max sequence length')
    parser.add_argument('--pooling_mode', type=str, default='mean',
                        choices=['mean', 'multiscale', 'final'])
    
    # Filtering options
    parser.add_argument('--bprp0_min', type=float, default=None,
                        help='Min BPRP0 cut')
    parser.add_argument('--bprp0_max', type=float, default=None,
                        help='Max BPRP0 cut')
    parser.add_argument('--require_age', action='store_true',
                        help='Only predict for stars with known ages (for validation)')
    
    # Ensemble options
    parser.add_argument('--ensemble', type=str, default='mean',
                        choices=['mean', 'median'],
                        help='How to combine fold predictions')
    
    # Processing
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for latent extraction')
    
    args = parser.parse_args()
    
    # Validation
    if args.model_type == 'latent' and args.rnn_model is None:
        parser.error("--rnn_model is required for latent model type")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load saved k-fold models
    print(f"\nLoading saved models from {args.model_file}")
    checkpoint = torch.load(args.model_file, map_location=device, weights_only=False)
    n_folds = checkpoint['n_folds']
    print(f"  Loaded {n_folds} fold models")
    
    # Load data CSV
    print(f"\nLoading data from {args.age_csv}")
    df = pd.read_csv(args.age_csv)
    
    # Apply filters
    valid_mask = ~df['BPRP0'].isna()
    
    if args.model_type == 'gyro':
        valid_mask &= ~df['Prot'].isna() & (df['Prot'] > 0)
    
    if args.bprp0_min is not None:
        valid_mask &= (df['BPRP0'] >= args.bprp0_min)
    if args.bprp0_max is not None:
        valid_mask &= (df['BPRP0'] <= args.bprp0_max)
    
    if args.require_age:
        valid_mask &= ~df['age_Myr'].isna()
    
    df_valid = df[valid_mask].copy()
    print(f"  {len(df_valid)} valid samples after filtering")
    
    # Extract data
    tic_ids = df_valid['TIC_ID'].values
    bprp0 = df_valid['BPRP0'].values
    
    if args.model_type == 'latent':
        # Load RNN and extract latent vectors
        print(f"\nLoading RNN model from {args.rnn_model}")
        rnn_model = load_rnn_model(
            args.rnn_model, args.hidden_size, args.direction,
            args.mode, args.use_flow, device
        )
        
        print(f"\nExtracting latent vectors...")
        latent_vectors, extracted_tics = extract_latent_vectors(
            rnn_model, args.h5_path, args.pickle_path, args.max_length,
            args.pooling_mode, args.batch_size, device, tic_ids_needed=tic_ids
        )
        
        # Match TIC IDs (some might be missing from H5)
        extracted_set = set(extracted_tics)
        keep_mask = np.array([tid in extracted_set for tid in tic_ids])
        
        if keep_mask.sum() < len(tic_ids):
            print(f"  Warning: {len(tic_ids) - keep_mask.sum()} TIC IDs not found in H5 file")
            df_valid = df_valid[keep_mask]
            tic_ids = tic_ids[keep_mask]
            bprp0 = bprp0[keep_mask]
        
        # Reorder latent vectors to match df order
        tic_to_idx = {tid: i for i, tid in enumerate(extracted_tics)}
        order = [tic_to_idx[tid] for tid in tic_ids]
        latent_vectors = latent_vectors[order]
        
        print(f"  Latent shape: {latent_vectors.shape}")
        
        # Predict
        print(f"\nPredicting ages with {n_folds}-fold ensemble ({args.ensemble})...")
        predictions, pred_std = predict_with_latent_model(
            checkpoint, latent_vectors, bprp0, device, args.ensemble
        )
    
    else:  # gyro
        prot = df_valid['Prot'].values
        
        print(f"\nPredicting ages with {n_folds}-fold ensemble ({args.ensemble})...")
        predictions, pred_std = predict_with_gyro_model(
            checkpoint, bprp0, prot, device, args.ensemble
        )
    
    # Build output dataframe
    output_df = pd.DataFrame({
        'TIC_ID': tic_ids,
        'predicted_age_myr': predictions,
        'prediction_std_myr': pred_std,
        'bprp0': bprp0,
    })
    
    if args.model_type == 'gyro':
        output_df['prot_days'] = df_valid['Prot'].values
    
    # Add true age if available
    if 'age_Myr' in df_valid.columns:
        true_ages = df_valid['age_Myr'].values
        output_df['true_age_myr'] = true_ages
        
        # Compute metrics for stars with known ages
        known_mask = ~np.isnan(true_ages)
        if known_mask.sum() > 0:
            mae = np.mean(np.abs(predictions[known_mask] - true_ages[known_mask]))
            corr = np.corrcoef(predictions[known_mask], true_ages[known_mask])[0, 1]
            log_scatter = np.std(np.log10(predictions[known_mask]) - np.log10(true_ages[known_mask]))
            
            print(f"\n=== Validation Metrics ({known_mask.sum()} stars with known ages) ===")
            print(f"MAE: {mae:.1f} Myr")
            print(f"Log scatter: {log_scatter:.3f} dex")
            print(f"Correlation: {corr:.3f}")
    
    # Save
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(output_df)} predictions to {output_path}")
    
    # Summary stats
    print(f"\nPrediction Summary:")
    print(f"  Min age: {predictions.min():.1f} Myr")
    print(f"  Max age: {predictions.max():.1f} Myr")
    print(f"  Median age: {np.median(predictions):.1f} Myr")
    print(f"  Mean ensemble std: {pred_std.mean():.1f} Myr")


if __name__ == '__main__':
    main()
