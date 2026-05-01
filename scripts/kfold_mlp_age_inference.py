"""K-fold age regression for exoplanet hosts using a simple MLP.

Conceptually parallel to kfold_age_inference.py but replaces the normalizing-flow
posterior with a plain MLP point-estimator that regresses log10(age_Myr) from the
encoder latent (and optional metadata: BPRP0, log10(BPRP0_err), [log10(MG)]).

Pipeline:
    1. Load cached latents for hosts (kfold-format or predict_ages format).
       Missing ages/BPRP0 are filled from age + metadata CSVs by GaiaDR3_ID.
    2. Per-sector → per-star aggregation (latent_mean/median/max/mean_std).
    3. Stratified K-fold by log-age. Per-fold: standardize → train MLP → predict.
    4. Aggregate predictions, write CSV + metrics + plots.
    5. Optionally train a deployment model on all stars (--train_full).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_host_latents(cache_path: str, age_csv: str, metadata_csv: str | None):
    """Load host latents and join ages/BPRP0/BPRP0_err/MG from CSVs by GaiaDR3_ID.

    Accepts both cache schemas:
      - kfold/UMAP: latent_vectors, gaia_ids, [ages, bprp0, ...]
      - predict_ages: latents, gaia_ids, bprp0, [bprp0_err]
    """
    d = np.load(cache_path, allow_pickle=True)
    keys = set(d.keys())
    lv_key = 'latent_vectors' if 'latent_vectors' in keys else 'latents'
    latents = d[lv_key]
    gaia_ids = d['gaia_ids'].astype(str)
    n = len(gaia_ids)
    print(f'Loaded latents cache ← {cache_path} ({n} samples, latent_dim={latents.shape[1]})')

    df_age = pd.read_csv(age_csv)
    df_age['GaiaDR3_ID'] = df_age['GaiaDR3_ID'].astype(str)
    if 'st_age' in df_age.columns and 'age_Myr' not in df_age.columns:
        df_age['age_Myr'] = df_age['st_age'].astype(float) * 1000.0  # Gyr → Myr

    if metadata_csv is not None:
        df_meta = pd.read_csv(metadata_csv)
        df_meta['GaiaDR3_ID'] = df_meta['GaiaDR3_ID'].astype(str)
        df = df_age.merge(df_meta, on='GaiaDR3_ID', how='left', suffixes=('', '_meta'))
    else:
        df = df_age

    def lookup(col, dtype=float):
        if col not in df.columns:
            return np.full(n, np.nan, dtype=dtype)
        m = dict(zip(df['GaiaDR3_ID'], df[col]))
        return np.array([m.get(g, np.nan) for g in gaia_ids], dtype=dtype)

    ages = lookup('age_Myr')
    # Take cache values when present, then NaN-fill from CSV. Caches written
    # before the host H5 Gaia ID repair have NaN BPRP0/BPRP0_err for ~18k rows
    # because the bad sci-notation IDs failed the original CSV lookup.
    bprp0 = d['bprp0'].astype(float) if 'bprp0' in keys else lookup('BPRP0')
    bprp0_err = d['bprp0_err'].astype(float) if 'bprp0_err' in keys else lookup('BPRP0_err')
    csv_b = lookup('BPRP0')
    csv_be = lookup('BPRP0_err')
    n_filled_b = int(((np.isnan(bprp0)) & (~np.isnan(csv_b))).sum())
    n_filled_be = int(((np.isnan(bprp0_err)) & (~np.isnan(csv_be))).sum())
    bprp0 = np.where(np.isnan(bprp0), csv_b, bprp0)
    bprp0_err = np.where(np.isnan(bprp0_err), csv_be, bprp0_err)
    if n_filled_b or n_filled_be:
        print(f'  filled NaN from CSV: BPRP0={n_filled_b}  BPRP0_err={n_filled_be}')
    mg = lookup('MG_quick')

    n_with_age = int(np.sum(~np.isnan(ages)))
    print(f'  ages joined: {n_with_age}/{n} ({100*n_with_age/max(n,1):.1f}%)')
    return latents, ages, bprp0, bprp0_err, mg, gaia_ids


def aggregate_by_star(latents, ages, bprp0, bprp0_err, mg, gaia_ids, method: str):
    """Reduce per-sector rows to one row per unique GaiaDR3_ID."""
    unique = np.unique(gaia_ids)
    out_lat, out_age, out_b, out_be, out_mg = [], [], [], [], []
    for gid in unique:
        mask = gaia_ids == gid
        L = latents[mask]
        if method == 'latent_mean':
            out_lat.append(L.mean(axis=0))
        elif method == 'latent_median':
            out_lat.append(np.median(L, axis=0))
        elif method == 'latent_max':
            out_lat.append(L.max(axis=0))
        elif method == 'latent_mean_std':
            out_lat.append(np.concatenate([L.mean(axis=0), L.std(axis=0)]))
        else:
            raise ValueError(f'Unknown star_aggregation: {method!r}')
        out_age.append(ages[mask][0])
        out_b.append(bprp0[mask][0])
        out_be.append(bprp0_err[mask][0])
        out_mg.append(mg[mask][0])
    print(f'{method}: reduced to {len(unique)} stars | dim={np.stack(out_lat).shape[1]}')
    return (np.stack(out_lat), np.array(out_age), np.array(out_b),
            np.array(out_be), np.array(out_mg), unique)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AgeRegressorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_features(latents, bprp0, bprp0_err, mg, use_metadata: bool, use_mg: bool):
    """Concat latents with optional metadata (BPRP0, log10(BPRP0_err), log10(MG))."""
    if not use_metadata:
        return latents.astype(np.float32)
    feats = [latents, bprp0[:, None]]
    with np.errstate(invalid='ignore', divide='ignore'):
        feats.append(np.log10(np.maximum(bprp0_err, 1e-6))[:, None])
        if use_mg:
            feats.append(np.log10(np.maximum(mg, 1e-6))[:, None])
    return np.concatenate(feats, axis=1).astype(np.float32)


def standardize(X_train, X_val):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


def train_one_fold(X_train, y_train, X_val, y_val, *, hidden_dims, dropout,
                   lr, weight_decay, n_epochs, batch_size, patience, device,
                   loss_fn='mse', verbose=False):
    """Train an MLP on (X_train, y_train), early-stop on val, return best model + curves."""
    in_dim = X_train.shape[1]
    model = AgeRegressorMLP(in_dim, hidden_dims, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    crit = nn.MSELoss() if loss_fn == 'mse' else nn.SmoothL1Loss()

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    best_val = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_since_best = 0
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(Xt)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), yv).item()
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if verbose and (epoch % 25 == 0 or epoch == n_epochs - 1):
            print(f'    epoch {epoch:3d}: train={epoch_loss:.4f}  val={val_loss:.4f}  best={best_val:.4f}')

        if epochs_since_best >= patience:
            if verbose:
                print(f'    early stop at epoch {epoch} (best val={best_val:.4f})')
            break

    model.load_state_dict(best_state)
    return model, np.array(train_losses), np.array(val_losses), best_val


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        return model(Xt).cpu().numpy()


# ---------------------------------------------------------------------------
# K-fold orchestration
# ---------------------------------------------------------------------------

def run_kfold(features, log_ages, *, n_folds, n_age_bins, seed, train_kwargs,
              device, verbose=False):
    """Stratified K-fold by log-age bin. Returns predictions, fold assignments,
    per-fold metrics, and per-fold loss curves."""
    bins = np.digitize(log_ages, np.quantile(log_ages, np.linspace(0, 1, n_age_bins + 1)[1:-1]))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    n = len(log_ages)
    predictions = np.full(n, np.nan)
    fold_assignments = np.full(n, -1, dtype=int)
    fold_models, fold_norms, fold_metrics, all_curves = [], [], [], []

    for fold_idx, (tr, va) in enumerate(skf.split(features, bins)):
        print(f'\n=== Fold {fold_idx + 1}/{n_folds}  train={len(tr)}  val={len(va)} ===')
        X_tr_raw, X_va_raw = features[tr], features[va]
        y_tr, y_va = log_ages[tr], log_ages[va]
        X_tr, X_va, mean, std = standardize(X_tr_raw, X_va_raw)
        t0 = time.time()
        model, tr_curve, va_curve, best_val = train_one_fold(
            X_tr, y_tr, X_va, y_va, device=device, verbose=verbose, **train_kwargs)
        elapsed = time.time() - t0
        y_pred = predict(model, X_va, device)
        mae = float(np.mean(np.abs(y_pred - y_va)))
        rmse = float(np.sqrt(np.mean((y_pred - y_va) ** 2)))
        r = float(pearsonr(y_pred, y_va)[0]) if len(y_va) > 2 else float('nan')
        print(f'  fold {fold_idx + 1}: MAE={mae:.4f} RMSE={rmse:.4f} r={r:.3f} '
              f'best_val={best_val:.4f} ({elapsed:.1f}s)')

        predictions[va] = y_pred
        fold_assignments[va] = fold_idx
        fold_models.append(model)
        fold_norms.append((mean, std))
        fold_metrics.append({'fold': fold_idx, 'mae': mae, 'rmse': rmse, 'pearson_r': r,
                             'best_val_loss': best_val, 'n_train': len(tr), 'n_val': len(va)})
        all_curves.append({'train': tr_curve, 'val': va_curve})

    return predictions, fold_assignments, fold_metrics, fold_models, fold_norms, all_curves


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(true_log, pred_log, output_dir: Path):
    valid = ~np.isnan(pred_log)
    t, p = true_log[valid], pred_log[valid]
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    r = float(pearsonr(p, t)[0]) if len(t) > 2 else float('nan')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(t, p, alpha=0.4, s=20)
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, lw=1)
    ax.set_xlabel('True log10(age / Myr)')
    ax.set_ylabel('Predicted log10(age / Myr)')
    ax.set_title(f'K-fold MLP age inference\nMAE={mae:.3f}  RMSE={rmse:.3f}  r={r:.3f}  n={len(t)}')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = output_dir / 'scatter_pred_vs_true.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
    return {'mae': mae, 'rmse': rmse, 'pearson_r': r, 'n': int(len(t))}


def plot_curves(all_curves, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, c in enumerate(all_curves):
        axes[0].plot(c['train'], alpha=0.6, label=f'fold {i}' if i < 5 else None)
        axes[1].plot(c['val'],   alpha=0.6, label=f'fold {i}' if i < 5 else None)
    for ax, t in zip(axes, ['Train loss', 'Val loss']):
        ax.set_xlabel('epoch'); ax.set_ylabel(t); ax.set_title(t); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = output_dir / 'training_curves.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--load_latents', type=str, required=True,
                        help='Path to host latents npz (kfold or predict_ages format).')
    parser.add_argument('--host_age_csv', type=str, required=True)
    parser.add_argument('--host_metadata_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--star_aggregation', type=str, default='latent_max',
                        choices=['latent_mean', 'latent_median', 'latent_max', 'latent_mean_std'])
    parser.add_argument('--use_metadata', action='store_true',
                        help='Concat BPRP0 + log10(BPRP0_err) (+ log10(MG) if --use_mg) to latents.')
    parser.add_argument('--use_mg', action='store_true')

    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--n_age_bins', type=int, default=10,
                        help='Bins used for age-stratified k-fold splitting.')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--mlp_hidden', type=int, nargs='+', default=[256, 128, 64])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'huber'])

    parser.add_argument('--train_full', action='store_true',
                        help='After k-fold, train one model on all stars and save full_model.pt.')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1. Load + aggregate
    latents, ages, bprp0, bprp0_err, mg, gaia_ids = load_host_latents(
        args.load_latents, args.host_age_csv, args.host_metadata_csv)

    # Drop any rows missing age, BPRP0, or BPRP0_err (we need them as features/target)
    valid = ~np.isnan(ages) & ~np.isnan(bprp0) & ~np.isnan(bprp0_err)
    if args.use_mg:
        valid &= ~np.isnan(mg)
    n_drop = int(np.sum(~valid))
    if n_drop > 0:
        print(f'Dropping {n_drop}/{len(ages)} rows with NaN in required fields')
    latents, ages, bprp0, bprp0_err, mg, gaia_ids = (
        latents[valid], ages[valid], bprp0[valid], bprp0_err[valid], mg[valid], gaia_ids[valid])

    star_lat, star_age, star_bprp0, star_berr, star_mg, star_gids = aggregate_by_star(
        latents, ages, bprp0, bprp0_err, mg, gaia_ids, method=args.star_aggregation)

    features = build_features(star_lat, star_bprp0, star_berr, star_mg,
                              use_metadata=args.use_metadata, use_mg=args.use_mg)
    log_ages = np.log10(star_age).astype(np.float32)
    print(f'Final dataset: {len(features)} stars | feature dim: {features.shape[1]}')

    # 2. K-fold
    train_kwargs = dict(
        hidden_dims=args.mlp_hidden, dropout=args.dropout,
        lr=args.lr, weight_decay=args.weight_decay,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        patience=args.patience, loss_fn=args.loss_fn,
    )
    predictions, fold_assignments, fold_metrics, fold_models, fold_norms, all_curves = run_kfold(
        features, log_ages,
        n_folds=args.n_folds, n_age_bins=args.n_age_bins, seed=args.seed,
        train_kwargs=train_kwargs, device=device, verbose=args.verbose)

    # 3. Plots + metrics
    overall = plot_results(log_ages, predictions, output_dir)
    plot_curves(all_curves, output_dir)

    # 4. Predictions CSV
    df_out = pd.DataFrame({
        'GaiaDR3_ID': star_gids,
        'true_log10_age_Myr': log_ages,
        'pred_log10_age_Myr': predictions,
        'true_age_Myr': star_age,
        'pred_age_Myr': 10 ** predictions,
        'BPRP0': star_bprp0,
        'BPRP0_err': star_berr,
        'fold': fold_assignments,
    })
    pred_path = output_dir / 'predictions.csv'
    df_out.to_csv(pred_path, index=False)
    print(f'Saved {pred_path}')

    # 5. Metrics + config
    summary = {
        'overall': overall,
        'per_fold': fold_metrics,
        'config': vars(args),
        'feature_dim': int(features.shape[1]),
        'n_stars': int(len(features)),
    }
    with (output_dir / 'metrics.json').open('w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'Saved {output_dir/"metrics.json"}')

    # 6. Save fold models + normalizers
    torch.save({
        'fold_state_dicts': [m.state_dict() for m in fold_models],
        'fold_norms': fold_norms,
        'config': vars(args),
        'feature_dim': features.shape[1],
    }, output_dir / 'kfold_models.pt')
    print(f'Saved {output_dir/"kfold_models.pt"}')

    # 7. Optional: train deployment model on all stars
    if args.train_full:
        print('\n=== Training deployment model on ALL stars ===')
        X_all, _, mean, std = standardize(features, features)  # 2nd return unused
        # Hold out a small slice for early stopping (random 10%)
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(X_all))
        n_val = max(1, len(X_all) // 10)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        model, _, _, best_val = train_one_fold(
            X_all[tr_idx], log_ages[tr_idx], X_all[val_idx], log_ages[val_idx],
            device=device, verbose=args.verbose, **train_kwargs)
        torch.save({
            'state_dict': model.state_dict(),
            'mean': mean, 'std': std,
            'config': vars(args),
            'feature_dim': features.shape[1],
        }, output_dir / 'full_model.pt')
        print(f'Saved {output_dir/"full_model.pt"}  (held-out val={best_val:.4f})')

    print('\nDone.')
    print(f'Overall: MAE={overall["mae"]:.4f}  RMSE={overall["rmse"]:.4f}  '
          f'r={overall["pearson_r"]:.3f}  n={overall["n"]}')


if __name__ == '__main__':
    main()
