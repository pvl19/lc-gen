"""K-fold cross-validation for age prediction from latent space + BPRP0.

This script trains k separate models, each on (k-1)/k of the data, and generates
age predictions for the held-out fold. The result is an unbiased age prediction
for every star, made by a model that never saw that star during training.

Star aggregation modes (--star_aggregation):
  none             - One sample per LC/sector; kfold splits by sector (legacy, has leakage
                     for multi-sector stars). Use only for backward compatibility.
  predict_mean     - One sample per LC/sector; kfold splits by *star* (no leakage);
                     per-sector predictions are averaged per star before reporting metrics.
  latent_mean      - Per-sector latent vectors are mean-pooled per star before kfold.
  latent_median    - Per-sector latent vectors are median-pooled per star before kfold.
  cross_sector     - Hidden states from all sectors of a star are concatenated before
                     multiscale pooling, yielding one latent vector per star.

Usage:
    python scripts/kfold_age_inference.py \\
        --model_path output/performance_tests/cf_final/model.pt \\
        --output_dir output/age_predictor/kfold \\
        --star_aggregation cross_sector \\
        --n_folds 10
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Ensure local 'src' is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_fresh(model_path: str, h5_path: str, csv_path: str,
                    hidden_size: int, direction: str, mode: str, use_flow: bool,
                    max_length: int, batch_size: int, pooling_mode: str,
                    max_samples: int = None, stratify_by_age: bool = False,
                    n_age_bins: int = 20, bprp0_min: float = None, bprp0_max: float = None,
                    use_metadata: bool = False, use_conv_channels: bool = False,
                    conv_config: dict = None, require_prot: bool = False,
                    device: torch.device = None):
    """Load model and extract per-sector latent vectors with all metadata.

    Returns:
        latent_vectors: (N, D) per-sector latent vectors
        ages:           (N,) stellar ages in Myr
        bprp0:          (N,) BP-RP0 colour
        gaia_ids:       (N,) GaiaDR3_ID strings
        tic_ids:        (N,) TESS TIC IDs (int64)
        sectors:        (N,) TESS sector numbers (int64)
        model:          loaded model (reused for cross_sector extraction)
    """
    from plot_umap_latent import (
        load_model, extract_latent_vectors, load_ages, get_stratified_indices
    )
    from lcgen.models.TimeSeriesDataset import METADATA_FEATURES

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Step 1: load full metadata to build the valid-sample mask ----------
    all_ages, all_bprp0, all_gaia_ids, _, _ = load_ages(h5_path, csv_path, sample_indices=None)

    valid_mask = ~np.isnan(all_ages) & ~np.isnan(all_bprp0)
    if bprp0_min is not None:
        valid_mask &= (all_bprp0 >= bprp0_min)
    if bprp0_max is not None:
        valid_mask &= (all_bprp0 <= bprp0_max)

    if require_prot:
        df_csv = pd.read_csv(csv_path)
        df_csv['GaiaDR3_ID'] = df_csv['GaiaDR3_ID'].astype(str)
        id_to_prot = dict(zip(df_csv['GaiaDR3_ID'], df_csv['Prot']))
        all_prot = np.array([id_to_prot.get(g, np.nan) for g in all_gaia_ids])
        valid_mask &= (~np.isnan(all_prot) & (all_prot > 0))
        print(f'Filtering to stars with rotation periods: {valid_mask.sum()} valid samples')

    # -- Step 2: determine sample indices -----------------------------------
    if max_samples is not None:
        if stratify_by_age:
            ages_for_strat = np.where(valid_mask, all_ages, np.nan)
            sample_indices = get_stratified_indices(ages_for_strat, max_samples, n_age_bins)
        else:
            sample_indices = np.where(valid_mask)[0][:max_samples]
    else:
        sample_indices = np.where(valid_mask)[0]

    # Guard against out-of-bounds indices
    with h5py.File(h5_path, 'r') as f:
        h5_size = len(f['flux'])
    ok = sample_indices < h5_size
    if not np.all(ok):
        print(f'Warning: {(~ok).sum()} indices exceed H5 size — dropped.')
        sample_indices = sample_indices[ok]

    # -- Step 3: load model and extract per-sector latents ------------------
    num_meta_features = len(METADATA_FEATURES) if use_metadata else 0
    model = load_model(model_path, device,
                       hidden_size=hidden_size, direction=direction, mode=mode,
                       use_flow=use_flow, num_meta_features=num_meta_features,
                       use_conv_channels=use_conv_channels, conv_config=conv_config)

    # Use the model's actual flags (load_model may have corrected them from the state dict)
    actual_use_metadata    = model.num_meta_features > 0
    actual_use_conv        = model.use_conv_channels

    latent_vectors = extract_latent_vectors(
        model, h5_path, device, max_length, batch_size, pooling_mode, sample_indices,
        use_metadata=actual_use_metadata, use_conv_channels=actual_use_conv
    )

    # -- Step 4: load metadata for selected samples -------------------------
    ages, bprp0, gaia_ids, tic_ids, sectors = load_ages(h5_path, csv_path, sample_indices)

    # Final validity filter (should all be valid, but defensive)
    final_ok = ~np.isnan(ages) & ~np.isnan(bprp0)
    latent_vectors = latent_vectors[final_ok]
    ages     = ages[final_ok]
    bprp0    = bprp0[final_ok]
    gaia_ids = gaia_ids[final_ok]
    tic_ids  = tic_ids[final_ok]
    sectors  = sectors[final_ok]

    print(f'Loaded {len(ages)} LC samples | latent dim: {latent_vectors.shape[1]}')
    print(f'Age range: {ages.min():.1f} – {ages.max():.1f} Myr')
    print(f'Unique stars: {len(np.unique(tic_ids))}')

    return latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors, sample_indices, model


# ---------------------------------------------------------------------------
# Latents cache  (save once, load many times)
# ---------------------------------------------------------------------------

def save_latents_cache(cache_path: str,
                       latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors,
                       cross_sector_latents=None, cross_sector_tic_ids=None):
    """Save per-sector latents + identifiers (and optionally cross-sector latents) to npz.

    The cache contains everything needed for all star_aggregation modes so that
    the model never needs to be re-run when comparing methods.

    Layout
    ------
    latent_vectors        (N, D)          per-sector pooled latents
    ages                  (N,)            stellar age in Myr
    bprp0                 (N,)            BP-RP0
    gaia_ids              (N,)  str       GaiaDR3_ID
    tic_ids               (N,)  int64     TESS TIC
    sectors               (N,)  int64     TESS sector number
    cross_sector_latents  (S, D)          one multiscale latent per unique star  [optional]
    cross_sector_tic_ids  (S,)  int64     TIC ordering for cross_sector_latents  [optional]
    """
    save_dict = dict(
        latent_vectors=latent_vectors,
        ages=ages, bprp0=bprp0,
        gaia_ids=gaia_ids.astype(str),
        tic_ids=tic_ids, sectors=sectors,
    )
    if cross_sector_latents is not None:
        save_dict['cross_sector_latents'] = cross_sector_latents
        save_dict['cross_sector_tic_ids'] = cross_sector_tic_ids
    np.savez(cache_path, **save_dict)
    keys = list(save_dict.keys())
    print(f'Saved latents cache → {cache_path}  (keys: {keys})')


def load_latents_cache(cache_path: str):
    """Load a latents cache written by save_latents_cache.

    Returns
    -------
    latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors,
    cross_sector_latents (or None), cross_sector_tic_ids (or None)
    """
    data = np.load(cache_path, allow_pickle=True)
    cross_sector_latents  = data['cross_sector_latents']  if 'cross_sector_latents' in data else None
    cross_sector_tic_ids  = data['cross_sector_tic_ids']  if 'cross_sector_tic_ids'  in data else None
    print(f'Loaded latents cache ← {cache_path}')
    print(f'  per-sector samples : {len(data["ages"])}')
    print(f'  latent dim         : {data["latent_vectors"].shape[1]}')
    if cross_sector_latents is not None:
        print(f'  cross-sector stars : {len(cross_sector_tic_ids)}')
    else:
        print(f'  cross-sector latents: not in cache')
    return (data['latent_vectors'], data['ages'], data['bprp0'],
            data['gaia_ids'], data['tic_ids'], data['sectors'],
            cross_sector_latents, cross_sector_tic_ids)


# ---------------------------------------------------------------------------
# Star-level aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_by_star(latent_vectors, ages, bprp0, tic_ids, method: str):
    """Reduce per-sector arrays to one row per unique star.

    Args:
        method: one of 'latent_mean', 'latent_median', 'latent_max',
                'latent_mean_std'.
                'latent_mean_std' concatenates [mean, std] along the feature
                axis, doubling the latent dimension. For single-sector stars
                the std is all zeros — the MLP sees this as a valid input but
                it carries no sector-variability information. Be aware this may
                introduce a confound if the number of sectors observed correlates
                with stellar properties.

    Returns:
        star_latents:    (n_stars, D)  — D is 2× input dim for latent_mean_std
        star_ages:       (n_stars,)
        star_bprp0:      (n_stars,)
        unique_tic_ids:  (n_stars,)
    """
    unique_tics = np.unique(tic_ids)
    star_latents = []
    star_ages    = []
    star_bprp0   = []

    for tic in unique_tics:
        mask = tic_ids == tic
        sectors_latents = latent_vectors[mask]  # (n_sectors, D)
        if method == 'latent_mean':
            star_latents.append(sectors_latents.mean(axis=0))
        elif method == 'latent_median':
            star_latents.append(np.median(sectors_latents, axis=0))
        elif method == 'latent_max':
            star_latents.append(sectors_latents.max(axis=0))
        elif method == 'latent_mean_std':
            mean = sectors_latents.mean(axis=0)
            std  = sectors_latents.std(axis=0)   # zero for single-sector stars
            star_latents.append(np.concatenate([mean, std]))
        else:
            raise ValueError(f'Unknown aggregation method: {method}')
        star_ages.append(ages[mask][0])
        star_bprp0.append(bprp0[mask][0])

    star_latents = np.stack(star_latents)
    print(f'{method}: reduced to {len(unique_tics)} stars | latent dim: {star_latents.shape[1]}')
    return star_latents, np.array(star_ages), np.array(star_bprp0), unique_tics


def extract_cross_sector_latents(model, h5_path: str, device: torch.device,
                                  sample_indices: np.ndarray, tic_ids_sorted: np.ndarray,
                                  max_length: int = 16384):
    """Pool hidden states across ALL sectors of each star via multiscale aggregation.

    Concatenates the valid hidden states from every sector of a star into a single
    sequence, then applies the same multiscale pooling used per-sector. This lets
    the model see the full temporal context of the star before collapsing to a
    single latent vector.

    Args:
        model:           trained BiDirectionalMinGRU
        h5_path:         path to H5 data file
        device:          torch device
        sample_indices:  H5 indices in sorted order (same ordering load_data_fresh uses)
        tic_ids_sorted:  TIC IDs aligned to sample_indices (same sorted order)
        max_length:      per-sector max sequence length

    Returns:
        star_latents:    (n_unique_stars, D)
        unique_tic_ids:  (n_unique_stars,)
    """
    from plot_umap_latent import compute_multiscale_features

    sorted_idx = np.sort(sample_indices)

    with h5py.File(h5_path, 'r') as f:
        flux_all     = f['flux'][sorted_idx]
        flux_err_all = f['flux_err'][sorted_idx]
        time_all     = f['time'][sorted_idx]
        lengths      = f['length'][sorted_idx]

    # Group positions in sorted arrays by TIC
    unique_tics      = np.unique(tic_ids_sorted)
    tic_to_positions = {t: np.where(tic_ids_sorted == t)[0] for t in unique_tics}

    star_latents = []
    model.eval()

    print(f'Cross-sector extraction for {len(unique_tics)} stars...')

    with torch.no_grad():
        for i, tic in enumerate(unique_tics):
            pos = tic_to_positions[tic]  # indices into sorted arrays

            flux_b = torch.tensor(flux_all[pos],     dtype=torch.float32, device=device)
            ferr_b = torch.tensor(flux_err_all[pos], dtype=torch.float32, device=device)
            time_b = torch.tensor(time_all[pos],     dtype=torch.float32, device=device)
            lens_b = lengths[pos]

            actual_max = min(max_length, flux_b.shape[1])
            flux_b = flux_b[:, :actual_max]
            ferr_b = ferr_b[:, :actual_max]
            time_b = time_b[:, :actual_max]
            lens_b = np.minimum(lens_b, actual_max)

            B, L = flux_b.shape
            mask = torch.zeros(B, L, device=device)
            for j, vlen in enumerate(lens_b):
                mask[j, :vlen] = 1.0

            x_in = torch.stack([flux_b, ferr_b], dim=-1)
            t_in = time_b.unsqueeze(-1)

            out   = model(x_in, t_in, mask=mask, return_states=True)
            h_fwd = out.get('h_fwd_tensor')
            h_bwd = out.get('h_bwd_tensor')

            # Collect valid hidden states from every sector of this star
            h_segments = []
            for j in range(B):
                vlen = lens_b[j]
                if h_fwd is not None and h_bwd is not None:
                    h = torch.cat([h_fwd[j, :vlen, :], h_bwd[j, :vlen, :]], dim=-1)
                elif h_fwd is not None:
                    h = h_fwd[j, :vlen, :]
                else:
                    h = h_bwd[j, :vlen, :]
                h_segments.append(h)

            h_all  = torch.cat(h_segments, dim=0)  # (total_valid_timesteps, hidden)
            latent = compute_multiscale_features(h_all, n_segments=4)
            star_latents.append(latent.cpu().numpy())

            if (i + 1) % 250 == 0:
                print(f'  {i + 1}/{len(unique_tics)} stars processed')

    star_latents = np.stack(star_latents)
    print(f'Cross-sector latents: {len(unique_tics)} stars | shape {star_latents.shape}')
    return star_latents, unique_tics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_single_fold(X_train, y_train, X_val, y_val,
                      hidden_dims, dropout, lr, weight_decay, n_epochs, batch_size, device):
    """Train one fold model, return validation predictions and best model state."""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=device),
        torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = AgePredictor(X_train.shape[1], hidden_dims, dropout).to(device)
    criterion  = nn.MSELoss()
    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_loss  = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
        epoch_loss /= len(X_train)
        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy().squeeze()

    return val_pred, best_loss, best_state


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------

def run_kfold_cv(latent_vectors, ages, bprp0, tic_ids,
                 n_folds=10, hidden_dims=[128, 64, 32], dropout=0.2,
                 lr=1e-3, weight_decay=1e-4, n_epochs=100, batch_size=64,
                 device=None, seed=42, save_models_dir=None,
                 star_level_split=False):
    """Run k-fold cross-validation.

    Args:
        star_level_split: if True, folds are assigned by unique TIC ID so all
                          sectors of a star always land in the same fold. This
                          avoids data leakage when the same star has multiple sectors.
                          Set to True for 'predict_mean'; irrelevant for aggregated
                          methods where there is already one row per star.

    Returns:
        predictions, true_ages, tic_ids, fold_assignments, fold_losses, norm_params
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(seed)
    n_samples = len(ages)

    # Normalise features
    bprp0_mean, bprp0_std = bprp0.mean(), bprp0.std()
    bprp0_norm = (bprp0 - bprp0_mean) / (bprp0_std + 1e-8)
    X = np.concatenate([latent_vectors, bprp0_norm[:, None]], axis=1)
    y = np.log10(ages)

    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    norm_params = {
        'X_mean': X_mean, 'X_std': X_std,
        'bprp0_mean': bprp0_mean, 'bprp0_std': bprp0_std,
        'latent_dim': latent_vectors.shape[1],
        'input_dim': X.shape[1],
    }

    # Build fold assignments
    fold_of_sample = np.zeros(n_samples, dtype=int)

    if star_level_split:
        unique_tics   = np.unique(tic_ids)
        perm          = np.random.permutation(len(unique_tics))
        tics_shuffled = unique_tics[perm]
        tic_fold_size = len(tics_shuffled) // n_folds
        tic_to_fold   = {}
        for f in range(n_folds):
            s = f * tic_fold_size
            e = s + tic_fold_size if f < n_folds - 1 else len(tics_shuffled)
            for t in tics_shuffled[s:e]:
                tic_to_fold[t] = f
        fold_of_sample = np.array([tic_to_fold[t] for t in tic_ids])
        print(f'Star-level split: {len(unique_tics)} unique stars across {n_folds} folds')
    else:
        perm      = np.random.permutation(n_samples)
        fold_size = n_samples // n_folds
        for f in range(n_folds):
            s = f * fold_size
            e = s + fold_size if f < n_folds - 1 else n_samples
            fold_of_sample[perm[s:e]] = f

    all_predictions = np.zeros(n_samples)
    fold_losses     = []
    fold_models     = []

    print(f'\nRunning {n_folds}-fold CV on {n_samples} samples '
          f'({"star-level" if star_level_split else "sample-level"} split)...')

    for fold in range(n_folds):
        val_idx   = np.where(fold_of_sample == fold)[0]
        train_idx = np.where(fold_of_sample != fold)[0]

        X_train, y_train = X_norm[train_idx], y[train_idx]
        X_val,   y_val   = X_norm[val_idx],   y[val_idx]

        val_pred_log, train_loss, model_state = train_single_fold(
            X_train, y_train, X_val, y_val,
            hidden_dims, dropout, lr, weight_decay, n_epochs, batch_size, device
        )
        fold_models.append(model_state)

        all_predictions[val_idx] = 10 ** val_pred_log
        fold_of_sample[val_idx]  = fold  # already set, but explicit
        fold_losses.append(train_loss)

        val_mae  = np.mean(np.abs(10 ** val_pred_log - ages[val_idx]))
        val_corr = np.corrcoef(10 ** val_pred_log, ages[val_idx])[0, 1]
        print(f'  Fold {fold + 1}/{n_folds}: train_loss={train_loss:.4f}  '
              f'val_MAE={val_mae:.1f} Myr  val_r={val_corr:.3f}')

    if save_models_dir is not None:
        save_dir = Path(save_models_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'fold_models': fold_models,
            'normalization_params': norm_params,
            'hidden_dims': hidden_dims,
            'dropout': dropout,
            'n_folds': n_folds,
            'seed': seed,
        }, save_dir / 'kfold_models.pt')
        print(f'Saved {n_folds} fold models to {save_dir / "kfold_models.pt"}')

    return all_predictions, ages, tic_ids, fold_of_sample, fold_losses, norm_params


# ---------------------------------------------------------------------------
# Results / plotting
# ---------------------------------------------------------------------------

def plot_kfold_results(predictions, true_ages, bprp0, tic_ids, fold_assignments, output_dir):
    """Create diagnostic plots and save metrics + predictions CSV."""
    mae       = np.mean(np.abs(predictions - true_ages))
    rmse      = np.sqrt(np.mean((predictions - true_ages) ** 2))
    log_mae   = np.mean(np.abs(np.log10(predictions) - np.log10(true_ages)))
    corr      = np.corrcoef(predictions, true_ages)[0, 1]
    med_ae    = np.median(np.abs(predictions - true_ages))
    log_resid = np.log10(predictions) - np.log10(true_ages)
    log_bias  = np.mean(log_resid)
    log_scat  = np.std(log_resid)

    print(f'\n=== K-Fold Results ===')
    print(f'N = {len(true_ages)}')
    print(f'MAE = {mae:.1f} Myr   Median AE = {med_ae:.1f} Myr   RMSE = {rmse:.1f} Myr')
    print(f'Log MAE = {log_mae:.3f} dex   bias = {log_bias:.3f} dex   scatter = {log_scat:.3f} dex')
    print(f'Pearson r = {corr:.3f}')

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    sc  = ax1.scatter(true_ages, predictions, c=bprp0, cmap='coolwarm',
                      alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    lo, hi = true_ages.min(), true_ages.max()
    ax1.plot([lo, hi], [lo, hi], 'k--', lw=2, label='Perfect')
    ax1.plot([lo, hi], [lo * 2, hi * 2], 'k:', alpha=0.5, label='±2×')
    ax1.plot([lo, hi], [lo / 2, hi / 2], 'k:', alpha=0.5)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('True Age (Myr)'); ax1.set_ylabel('Predicted Age (Myr)')
    ax1.set_title(f'K-Fold Age Predictions (r={corr:.3f})')
    ax1.legend(loc='upper left')
    plt.colorbar(sc, ax=ax1, label='BP-RP₀')
    ax1.text(0.02, 0.98,
             f'N={len(true_ages)}\nMAE={mae:.0f} Myr\nLog σ={log_scat:.2f} dex\nr={corr:.3f}',
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(true_ages, predictions - true_ages, c=bprp0, cmap='coolwarm',
                alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    ax2.axhline(0, color='k', linestyle='--', lw=2)
    ax2.set_xscale('log')
    ax2.set_xlabel('True Age (Myr)'); ax2.set_ylabel('Residual (Pred − True) [Myr]')
    ax2.set_title('Residuals vs True Age')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(log_resid, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='r', linestyle='--', lw=2, label='Zero')
    ax3.axvline(log_bias, color='orange', lw=2, label=f'Mean={log_bias:.2f}')
    ax3.set_xlabel('Log Residual (dex)'); ax3.set_ylabel('Count')
    ax3.set_title(f'Log Residuals (σ={log_scat:.2f} dex)')
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(bprp0, log_resid, c=np.log10(true_ages), cmap='viridis', alpha=0.6, s=15)
    ax4.axhline(0, color='k', linestyle='--', lw=2)
    ax4.set_xlabel('BP-RP₀'); ax4.set_ylabel('Log Residual (dex)')
    ax4.set_title('Residuals vs Colour (by log age)')
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(np.log10(true_ages).min(),
                                                   np.log10(true_ages).max()))
    plt.colorbar(sm, ax=ax4, label='log(Age/Myr)')

    plt.tight_layout()
    plt.savefig(output_dir / 'kfold_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {output_dir / "kfold_results.png"}')

    pd.DataFrame({
        'TIC_ID': tic_ids,
        'true_age_myr': true_ages,
        'predicted_age_myr': predictions,
        'bprp0': bprp0,
        'fold': fold_assignments,
        'log_residual': log_resid,
    }).to_csv(output_dir / 'kfold_predictions.csv', index=False)
    print(f'Saved predictions to {output_dir / "kfold_predictions.csv"}')

    metrics = {
        'n_samples': int(len(true_ages)),
        'mae_myr': float(mae), 'median_ae_myr': float(med_ae), 'rmse_myr': float(rmse),
        'log_mae_dex': float(log_mae), 'log_bias_dex': float(log_bias),
        'log_scatter_dex': float(log_scat), 'correlation': float(corr),
    }
    with open(output_dir / 'kfold_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='K-fold cross-validation for age prediction')

    # Data / model
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint. Required unless --load_latents is set.')
    parser.add_argument('--h5_path',    type=str, default='data/timeseries_x.h5')
    parser.add_argument('--age_csv',    type=str, default='data/phot_all.csv')
    parser.add_argument('--output_dir', type=str, default='output/age_predictor/kfold')

    # (kept for backward compat with old shell scripts; not used)
    parser.add_argument('--pickle_path', type=str, default=None,
                        help='Deprecated — ignored. Data is now loaded from --h5_path.')

    # Model extraction
    parser.add_argument('--hidden_size',      type=int,   default=64)
    parser.add_argument('--direction',        type=str,   default='bi',
                        choices=['forward', 'backward', 'bi'])
    parser.add_argument('--mode',             type=str,   default='parallel',
                        choices=['sequential', 'parallel'])
    parser.add_argument('--use_flow',         action='store_true')
    parser.add_argument('--use_metadata',     action='store_true')
    parser.add_argument('--use_conv_channels',action='store_true')
    parser.add_argument('--conv_encoder_type',type=str,   default='unet',
                        choices=['lightweight', 'unet'])
    parser.add_argument('--max_length',       type=int,   default=16384)
    parser.add_argument('--pooling_mode',     type=str,   default='multiscale',
                        choices=['mean', 'multiscale', 'final'])

    # Star aggregation
    parser.add_argument('--star_aggregation', type=str,   default='none',
                        choices=['none', 'predict_mean', 'latent_mean', 'latent_median',
                                 'latent_max', 'latent_mean_std', 'cross_sector'],
                        help=(
                            'none          – one sample per sector; kfold splits by sector '
                            '(legacy, has leakage for multi-sector stars). '
                            'predict_mean  – kfold splits by star; per-sector predictions '
                            'are averaged per star before metrics. '
                            'latent_mean   – sector latents are mean-pooled per star. '
                            'latent_median – sector latents are median-pooled per star. '
                            'cross_sector  – hidden states concatenated across sectors '
                            'before multiscale pooling.'
                        ))

    # Data selection
    parser.add_argument('--max_samples',     type=int,   default=None)
    parser.add_argument('--stratify_by_age', action='store_true')
    parser.add_argument('--n_age_bins',      type=int,   default=20)
    parser.add_argument('--bprp0_min',       type=float, default=None)
    parser.add_argument('--bprp0_max',       type=float, default=None)
    parser.add_argument('--require_prot',    action='store_true')

    # K-fold
    parser.add_argument('--n_folds',    type=int,   default=10)
    parser.add_argument('--seed',       type=int,   default=42)

    # MLP / training
    parser.add_argument('--hidden_dims',   type=int,   nargs='+', default=[128, 64, 32])
    parser.add_argument('--dropout',       type=float, default=0.2)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--n_epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',    type=int,   default=64)

    # Latents cache
    parser.add_argument('--save_latents', type=str, default=None, metavar='PATH',
                        help='After extracting latents, save them (+ cross-sector latents) '
                             'to this npz file. Pass to --load_latents on subsequent runs '
                             'to skip model inference entirely.')
    parser.add_argument('--load_latents', type=str, default=None, metavar='PATH',
                        help='Load pre-computed latents from this npz file instead of '
                             'running the model. --model_path is not required when this '
                             'flag is set.')

    args = parser.parse_args()

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device: {device}')

    conv_config = None
    if args.use_conv_channels:
        if args.conv_encoder_type == 'lightweight':
            conv_config = {'encoder_type': 'lightweight', 'hidden_channels': 16,
                           'num_layers': 3, 'activation': 'gelu'}
        else:
            conv_config = {'encoder_type': 'unet', 'input_length': 16000,
                           'encoder_dims': [4, 8, 16, 32], 'num_layers': 4,
                           'activation': 'gelu'}

    # ── Step 1: per-sector latents (from model or from cache) ──────────────
    cached_cross_sector_latents  = None
    cached_cross_sector_tic_ids  = None

    if args.load_latents:
        print(f'\n=== Loading latents from cache: {args.load_latents} ===')
        (latent_vectors, ages, bprp0, _, tic_ids, sectors,
         cached_cross_sector_latents, cached_cross_sector_tic_ids) = load_latents_cache(args.load_latents)
        model          = None
        sample_indices = None  # not needed when loading from cache
    else:
        if args.model_path is None:
            parser.error('--model_path is required unless --load_latents is set.')
        print('\n=== Extracting per-sector latent vectors ===')
        (latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors,
         sample_indices, model) = load_data_fresh(
            model_path=args.model_path,
            h5_path=args.h5_path,
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
            use_metadata=args.use_metadata,
            use_conv_channels=args.use_conv_channels,
            conv_config=conv_config,
            require_prot=args.require_prot,
            device=device,
        )

        if args.save_latents:
            # Pre-compute cross-sector latents while the model is loaded so that
            # cross_sector runs are also covered by a single cache file.
            print('\n=== Pre-computing cross-sector latents for cache ===')
            cached_cross_sector_latents, cached_cross_sector_tic_ids = extract_cross_sector_latents(
                model, args.h5_path, device,
                sample_indices=sample_indices,
                tic_ids_sorted=tic_ids,
                max_length=args.max_length,
            )
            save_latents_cache(
                args.save_latents,
                latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors,
                cached_cross_sector_latents, cached_cross_sector_tic_ids,
            )

    # ── Step 2: apply star aggregation ─────────────────────────────────────
    agg = args.star_aggregation
    star_level_split = False  # whether run_kfold_cv should split by unique TIC

    if agg == 'none':
        # Legacy: per-sector, kfold by sample (may have leakage)
        kfold_latents = latent_vectors
        kfold_ages    = ages
        kfold_bprp0   = bprp0
        kfold_tics    = tic_ids

    elif agg == 'predict_mean':
        # Per-sector latents + star-level kfold split; average preds per star after
        kfold_latents = latent_vectors
        kfold_ages    = ages
        kfold_bprp0   = bprp0
        kfold_tics    = tic_ids
        star_level_split = True

    elif agg in ('latent_mean', 'latent_median', 'latent_max', 'latent_mean_std'):
        print(f'\n=== Aggregating latents ({agg}) ===')
        kfold_latents, kfold_ages, kfold_bprp0, kfold_tics = aggregate_by_star(
            latent_vectors, ages, bprp0, tic_ids, method=agg
        )

    elif agg == 'cross_sector':
        print('\n=== Cross-sector hidden-state pooling ===')
        if cached_cross_sector_latents is not None:
            # Fast path: already computed (either from --save_latents or --load_latents)
            kfold_latents = cached_cross_sector_latents
            kfold_tics    = cached_cross_sector_tic_ids
        else:
            if model is None:
                raise RuntimeError(
                    'cross_sector requires either --load_latents (with pre-cached cross-sector '
                    'latents) or a model loaded via --model_path.'
                )
            kfold_latents, kfold_tics = extract_cross_sector_latents(
                model, args.h5_path, device,
                sample_indices=sample_indices,
                tic_ids_sorted=tic_ids,
                max_length=args.max_length,
            )
        tic_to_age   = {t: ages[tic_ids == t][0]  for t in np.unique(tic_ids)}
        tic_to_bprp0 = {t: bprp0[tic_ids == t][0] for t in np.unique(tic_ids)}
        kfold_ages  = np.array([tic_to_age[t]   for t in kfold_tics])
        kfold_bprp0 = np.array([tic_to_bprp0[t] for t in kfold_tics])

    else:
        raise ValueError(f'Unknown star_aggregation: {agg}')

    # ── Step 3: k-fold cross-validation ────────────────────────────────────
    print('\n=== Running K-Fold Cross-Validation ===')
    predictions, true_ages, result_tics, fold_assignments, fold_losses, _ = run_kfold_cv(
        latent_vectors=kfold_latents,
        ages=kfold_ages,
        bprp0=kfold_bprp0,
        tic_ids=kfold_tics,
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
        star_level_split=star_level_split,
    )

    # ── Step 4: for predict_mean, save per-sector CSV then average per star ─
    if agg == 'predict_mean':
        # Save per-sector predictions first — useful for uncertainty estimation
        # (spread of per-sector predictions = observational scatter per star)
        pd.DataFrame({
            'TIC_ID': result_tics,
            'sector': sectors,
            'true_age_myr': true_ages,
            'predicted_age_myr': predictions,
            'bprp0': kfold_bprp0,
            'fold': fold_assignments,
        }).to_csv(output_dir / 'kfold_predictions_per_sector.csv', index=False)
        print(f'Saved per-sector predictions to {output_dir / "kfold_predictions_per_sector.csv"}')

        print('\nAveraging per-sector predictions per star...')
        unique_tics = np.unique(result_tics)
        predictions    = np.array([predictions[result_tics == t].mean()  for t in unique_tics])
        true_ages      = np.array([true_ages[result_tics == t][0]        for t in unique_tics])
        kfold_bprp0    = np.array([kfold_bprp0[result_tics == t][0]      for t in unique_tics])
        fold_assignments = np.array([fold_assignments[result_tics == t][0] for t in unique_tics])
        result_tics    = unique_tics
        print(f'Averaged to {len(unique_tics)} unique stars')

    # ── Step 5: results ─────────────────────────────────────────────────────
    plot_kfold_results(
        predictions, true_ages, kfold_bprp0, result_tics, fold_assignments, output_dir
    )

    config = vars(args)
    config.update({
        'latent_dim':    int(kfold_latents.shape[1]),
        'n_samples':     int(len(kfold_ages)),
        'fold_losses':   [float(l) for l in fold_losses],
        'star_level_split': star_level_split,
    })
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print('\nDone!')


if __name__ == '__main__':
    main()
