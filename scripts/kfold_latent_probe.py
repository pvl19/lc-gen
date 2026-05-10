"""K-fold MLP probe: can the autoencoder latent space recover a per-light-curve
summary statistic?

Four probe targets:
    flux_skew  — asinh-transformed per-sector flux skewness
    flux_kurt  — asinh-transformed per-sector flux excess kurtosis
    lit_prot   — log10 literature rotation period (per-star, replicated per sector)
    tars_prot  — log10 TARS-pipeline rotation period (per-star, replicated per sector)

Baselines (--baseline):
    none      — real latents
    gaussian  — replace latents with iid N(0,1) of identical shape (pure-noise floor)
    shuffle   — permute rows of the real latent matrix (preserves marginals, kills mapping)

Splits: GroupKFold(k=5) on GaiaDR3_ID so all sectors of one star stay in one fold.
Metrics are per-sector (no per-star aggregation in this script).

Outputs (in --output_dir):
    metrics.json, predictions.csv, scatter_pred_vs_true.png, training_curves.png

Usage:
    python scripts/kfold_latent_probe.py \\
        --probe flux_skew \\
        --baseline none \\
        --latents final_model/parallel_fixed/e60/latents.npz \\
        --moments_csv final_pretrain/flux_moments.csv \\
        --combined_csv data/all_combined_metadata.csv \\
        --output_dir output/latent_probes/flux_skew/
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset


PROBE_TRANSFORMS = {
    'flux_skew': ('asinh',  np.arcsinh, np.sinh, 'asinh(flux_skew)'),
    'flux_kurt': ('asinh',  np.arcsinh, np.sinh, 'asinh(flux_kurt)'),
    'lit_prot':  ('log10',  np.log10,   lambda y: 10.0 ** y, 'log10(lit_Prot / d)'),
    'tars_prot': ('log10',  np.log10,   lambda y: 10.0 ** y, 'log10(tars_Prot / d)'),
}


def load_targets(probe: str, gaia_ids: np.ndarray, sectors: np.ndarray,
                 moments_csv: Path, combined_csv: Path) -> np.ndarray:
    """Return per-row target aligned to (gaia_ids, sectors). NaN where missing."""
    gids = pd.Series(gaia_ids).astype(str).values
    secs = sectors.astype(int)

    if probe in ('flux_skew', 'flux_kurt'):
        col = probe  # column name matches probe name
        df = pd.read_csv(moments_csv, usecols=['GaiaDR3_ID', 'sector', col])
        df['GaiaDR3_ID'] = df['GaiaDR3_ID'].astype(str)
        df = df.drop_duplicates(subset=['GaiaDR3_ID', 'sector'], keep='first')
        key = pd.DataFrame({'GaiaDR3_ID': gids, 'sector': secs})
        merged = key.merge(df, on=['GaiaDR3_ID', 'sector'], how='left')
        return merged[col].to_numpy(dtype=np.float64)

    if probe in ('lit_prot', 'tars_prot'):
        col = 'lit_Prot' if probe == 'lit_prot' else 'tars_Prot'
        df = pd.read_csv(combined_csv, usecols=['GaiaDR3_ID', col])
        df['GaiaDR3_ID'] = df['GaiaDR3_ID'].astype(str)
        df = df.drop_duplicates(subset='GaiaDR3_ID', keep='first')
        key = pd.DataFrame({'GaiaDR3_ID': gids})
        merged = key.merge(df, on='GaiaDR3_ID', how='left')
        y = merged[col].to_numpy(dtype=np.float64)
        # log10 requires strictly positive Prot
        y[~np.isfinite(y) | (y <= 0)] = np.nan
        return y

    raise ValueError(f'unknown probe {probe}')


def apply_baseline(X: np.ndarray, baseline: str, rng: np.random.Generator) -> np.ndarray:
    if baseline == 'none':
        return X
    if baseline == 'gaussian':
        return rng.standard_normal(X.shape).astype(np.float32)
    if baseline == 'shuffle':
        perm = rng.permutation(X.shape[0])
        return X[perm]
    raise ValueError(f'unknown baseline {baseline}')


class MLP(nn.Module):
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


def standardize(X_train, X_val):
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


def train_one_fold(X_tr, y_tr, X_va, y_va, *, hidden_dims, dropout, lr,
                   weight_decay, n_epochs, batch_size, patience, device):
    model = MLP(X_tr.shape[1], hidden_dims, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.MSELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=device)
    yv = torch.tensor(y_va, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    best_val = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    since_best = 0
    tr_curve, va_curve = [], []
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
        tr_curve.append(epoch_loss)
        va_curve.append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    return model, np.array(tr_curve), np.array(va_curve), best_val


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


def plot_scatter(true, pred, title_units: str, out: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true, pred, alpha=0.3, s=8)
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, lw=1)
    ax.set_xlabel(f'True {title_units}')
    ax.set_ylabel(f'Predicted {title_units}')
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    r = float(pearsonr(pred, true)[0]) if len(true) > 2 else float('nan')
    ax.set_title(f'MAE={mae:.4f}  RMSE={rmse:.4f}  r={r:.3f}  n={len(true)}')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_curves(curves, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (tr, va) in enumerate(curves):
        axes[0].plot(tr, alpha=0.6, label=f'fold {i}')
        axes[1].plot(va, alpha=0.6, label=f'fold {i}')
    for ax, t in zip(axes, ['Train loss', 'Val loss']):
        ax.set_xlabel('epoch'); ax.set_ylabel(t); ax.set_title(t); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probe', required=True, choices=list(PROBE_TRANSFORMS.keys()))
    ap.add_argument('--baseline', default='none', choices=['none', 'gaussian', 'shuffle'])
    ap.add_argument('--latents', default='final_model/parallel_fixed/e60/latents.npz')
    ap.add_argument('--moments_csv', default='final_pretrain/flux_moments.csv')
    ap.add_argument('--combined_csv', default='data/all_combined_metadata.csv')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64])
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--n_epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f'Probe: {args.probe}  baseline: {args.baseline}  device: {args.device}')

    # ---- load latents + metadata ----------------------------------------
    z = np.load(args.latents, allow_pickle=True)
    if 'latent_vectors' in z.files:
        X = z['latent_vectors'].astype(np.float32)
    else:
        X = z['latents'].astype(np.float32)
    gaia_ids = z['gaia_ids']
    sectors  = z['sectors'] if 'sectors' in z.files else np.zeros(len(X), dtype=np.int64)
    print(f'  latents: {X.shape}')

    # ---- load + transform targets ---------------------------------------
    y_raw = load_targets(args.probe, gaia_ids, sectors,
                         Path(args.moments_csv), Path(args.combined_csv))
    valid = np.isfinite(y_raw)
    print(f'  target {args.probe}: {valid.sum()}/{len(y_raw)} non-NaN rows')

    X = X[valid]; y_raw = y_raw[valid]
    gaia_ids = np.asarray(gaia_ids)[valid]
    sectors  = sectors[valid]

    _, fwd, inv, units = PROBE_TRANSFORMS[args.probe]
    y = fwd(y_raw).astype(np.float64)

    # ---- baseline injection ---------------------------------------------
    X = apply_baseline(X, args.baseline, rng)

    # ---- groupkfold by Gaia ID ------------------------------------------
    gkf = GroupKFold(n_splits=args.n_folds)
    n = len(y)
    predictions = np.full(n, np.nan)
    fold_assign = np.full(n, -1, dtype=int)
    fold_metrics, curves = [], []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=gaia_ids)):
        print(f'\n=== Fold {fold_idx + 1}/{args.n_folds}  train={len(tr_idx)}  val={len(va_idx)} ===')
        X_tr, X_va, _, _ = standardize(X[tr_idx], X[va_idx])
        y_tr_raw, y_va_raw = y[tr_idx], y[va_idx]
        y_mean = float(y_tr_raw.mean()); y_std = float(y_tr_raw.std() + 1e-8)
        y_tr = (y_tr_raw - y_mean) / y_std
        y_va = (y_va_raw - y_mean) / y_std

        t0 = time.time()
        model, tr_c, va_c, best_val = train_one_fold(
            X_tr.astype(np.float32), y_tr.astype(np.float32),
            X_va.astype(np.float32), y_va.astype(np.float32),
            hidden_dims=args.hidden_dims, dropout=args.dropout,
            lr=args.lr, weight_decay=args.weight_decay,
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            patience=args.patience, device=args.device,
        )
        y_pred_norm = predict(model, X_va.astype(np.float32), args.device)
        y_pred = y_pred_norm * y_std + y_mean

        mae = float(np.mean(np.abs(y_pred - y_va_raw)))
        rmse = float(np.sqrt(np.mean((y_pred - y_va_raw) ** 2)))
        r = float(pearsonr(y_pred, y_va_raw)[0]) if len(y_va_raw) > 2 else float('nan')
        print(f'  fold {fold_idx + 1}: MAE={mae:.4f} RMSE={rmse:.4f} r={r:.3f} '
              f'best_val={best_val:.4f} ({time.time() - t0:.1f}s)')

        predictions[va_idx] = y_pred
        fold_assign[va_idx] = fold_idx
        fold_metrics.append({'fold': fold_idx, 'mae': mae, 'rmse': rmse,
                             'pearson_r': r, 'best_val_loss': best_val,
                             'n_train': int(len(tr_idx)), 'n_val': int(len(va_idx))})
        curves.append((tr_c, va_c))

    # ---- overall metrics ------------------------------------------------
    mask = ~np.isnan(predictions)
    pt = predictions[mask]; tt = y[mask]
    overall = {
        'probe': args.probe,
        'baseline': args.baseline,
        'transform': PROBE_TRANSFORMS[args.probe][0],
        'mae_transformed':  float(np.mean(np.abs(pt - tt))),
        'rmse_transformed': float(np.sqrt(np.mean((pt - tt) ** 2))),
        'pearson_r_transformed': float(pearsonr(pt, tt)[0]) if len(tt) > 2 else float('nan'),
        'n': int(mask.sum()),
        'per_fold': fold_metrics,
    }
    pred_raw = inv(predictions)
    raw_true = y_raw
    mask_raw = mask & np.isfinite(pred_raw) & np.isfinite(raw_true)
    if mask_raw.sum() > 2:
        overall['mae_raw'] = float(np.mean(np.abs(pred_raw[mask_raw] - raw_true[mask_raw])))
        overall['pearson_r_raw'] = float(pearsonr(pred_raw[mask_raw], raw_true[mask_raw])[0])

    print('\nOverall (transformed): MAE={mae_transformed:.4f} RMSE={rmse_transformed:.4f} '
          'r={pearson_r_transformed:.3f}  n={n}'.format(**overall))

    # ---- save outputs ---------------------------------------------------
    (out_dir / 'metrics.json').write_text(json.dumps(overall, indent=2))
    pd.DataFrame({
        'GaiaDR3_ID': gaia_ids,
        'sector': sectors,
        'fold': fold_assign,
        'y_true_raw': y_raw,
        'y_pred_raw': pred_raw,
        'y_true_transformed': y,
        'y_pred_transformed': predictions,
    }).to_csv(out_dir / 'predictions.csv', index=False)
    plot_scatter(tt, pt, units, out_dir / 'scatter_pred_vs_true.png')
    plot_curves(curves, out_dir / 'training_curves.png')
    print(f'Saved outputs to {out_dir}')


if __name__ == '__main__':
    main()
