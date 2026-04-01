"""K-fold gyrochronology baseline: NLE flow modelling p(log10(Prot) | log10_age, BPRP0, log10(BPRP0_err)).

Mirrors the kfold_age_inference.py NLE framework exactly, but uses a single
observable (log10(Prot)) as the flow's feature variable instead of PCA-compressed
autoencoder latents. No PCA step is needed.

Usage:
    python scripts/kfold_gyro_baseline.py \
        --age_csv data/phot_all.csv \
        --output_dir output/age_predictor_tests/gyro_baseline_nle
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

try:
    import zuko
except ImportError:
    raise ImportError("zuko is required. Install with: pip install zuko")

# ---------------------------------------------------------------------------
# Constants (match kfold_age_inference.py)
# ---------------------------------------------------------------------------
PRIOR_LOGA_MYR      = (0.0, 4.14)   # log10(age/Myr) range for the uniform age prior
LOGA_GRID_DEFAULT_SIZE = 1000
P_CLUSTER_MEM       = 0.9            # default membership probability (used when mem_prob is NaN)
P_OUTLIER           = 0.05           # fraction of flow weight assigned to outlier component

# Outlier distribution: uniform over this log10(Prot/days) range
# Matches train_chronoflow.ipynb prior_prot = [-1.75, 2.5]
LOG_PROT_OUTLIER_MIN = -1.75   # log10(~0.018 days)
LOG_PROT_OUTLIER_MAX =  2.5    # log10(~316 days)
LOG_PROT_OUTLIER_LN_DENSITY = -np.log(LOG_PROT_OUTLIER_MAX - LOG_PROT_OUTLIER_MIN)  # ln(1/range)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GyroNLEPredictor(nn.Module):
    """NLE flow for gyrochronology: models p(log10(Prot) | log10_age, BPRP0, log10(BPRP0_err)).

    Identical architecture to AgePredictor in kfold_age_inference.py, but
    features=1 (scalar log10(Prot)) instead of a multi-dim PCA latent.
    """

    def __init__(self, flow_transforms: int = 8, flow_hidden_features: list = None):
        super().__init__()
        if flow_hidden_features is None:
            flow_hidden_features = [64, 64]

        # NSF: 1D feature (log10 Prot), 3D context (log10_age, bprp0, log10(bprp0_err))
        self.flow = zuko.flows.NSF(
            features=1,
            context=3,
            transforms=flow_transforms,
            hidden_features=flow_hidden_features,
        )

    def _nll_with_outlier(self, log_prob_flow: torch.Tensor, z: torch.Tensor,
                          mem_prob: torch.Tensor = None) -> torch.Tensor:
        if mem_prob is None:
            p_mem = P_CLUSTER_MEM
        else:
            p_mem = torch.where(torch.isnan(mem_prob),
                                torch.full_like(mem_prob, P_CLUSTER_MEM), mem_prob)
        nf_weight    = p_mem * (1.0 - P_OUTLIER)
        # Uniform outlier over the fixed log10(Prot) range — constant log density
        ln_p_outlier = torch.full_like(log_prob_flow, LOG_PROT_OUTLIER_LN_DENSITY)
        ln_p_combined = torch.logsumexp(
            torch.stack([
                torch.log(nf_weight)       + log_prob_flow,
                torch.log(1.0 - nf_weight) + ln_p_outlier,
            ], dim=0), dim=0
        )
        return -ln_p_combined.mean()

    def forward(self, z: torch.Tensor, log_age: torch.Tensor,
                bprp0: torch.Tensor, log_bprp0_err: torch.Tensor,
                mem_prob: torch.Tensor = None) -> torch.Tensor:
        """NLL training loss.

        z:            (B, 1) log10(Prot) — the likelihood variable
        log_age:      (B,)   log10(age/Myr) — flow context
        bprp0:        (B,)   raw BPRP0 — flow context
        log_bprp0_err:(B,)   log10(BPRP0_err) — flow context
        mem_prob:     (B,)   cluster membership probability (NaN → fallback to P_CLUSTER_MEM)
        """
        context = torch.stack([log_age, bprp0, log_bprp0_err], dim=1)
        return self._nll_with_outlier(self.flow(context).log_prob(z), z, mem_prob)

    def predict_stats(self, z: torch.Tensor, bprp0: torch.Tensor,
                      log_bprp0_err: torch.Tensor,
                      loga_grid: torch.Tensor) -> dict:
        """Grid-based posterior: p(age | Prot, bprp0) ∝ p(Prot | age, bprp0) · p(age).

        z:         (B, 1) log10(Prot)
        loga_grid: (G,)   log10(age/Myr) evaluation grid
        Returns dict of (B,) tensors: median, mean, map, p16, p84 — all log10(age/Myr).
        """
        B = z.shape[0]
        G = loga_grid.shape[0]

        z_exp     = z.unsqueeze(1).expand(B, G, 1).reshape(B * G, 1)
        bprp0_exp = bprp0.unsqueeze(1).expand(B, G).reshape(B * G)
        berr_exp  = log_bprp0_err.unsqueeze(1).expand(B, G).reshape(B * G)
        loga_exp  = loga_grid.unsqueeze(0).expand(B, G).reshape(B * G)
        context   = torch.stack([loga_exp, bprp0_exp, berr_exp], dim=1)

        log_lik   = self.flow(context).log_prob(z_exp).reshape(B, G)
        log_post  = log_lik - torch.logsumexp(log_lik, dim=1, keepdim=True)
        posterior = log_post.exp()

        map_idx = posterior.argmax(dim=1)
        map_est = loga_grid[map_idx]
        mean    = (posterior * loga_grid.unsqueeze(0)).sum(dim=1)
        cdf     = posterior.cumsum(dim=1)

        def pct(p: float) -> torch.Tensor:
            return loga_grid[(cdf >= p).float().argmax(dim=1)]

        return {
            'median': pct(0.5),
            'mean':   mean,
            'map':    map_est,
            'p16':    pct(0.16),
            'p84':    pct(0.84),
        }


# ---------------------------------------------------------------------------
# LR scheduler (identical to kfold_age_inference.py)
# ---------------------------------------------------------------------------

class CombinedLRScheduler:
    """Exponential decay × cosine annealing.

    lr = initial_lr * decay_rate^epoch * 0.5 * (1 + cos(π * epoch / T_max))
    """
    def __init__(self, optimizer, n_epochs: int, initial_lr: float,
                 decay_rate: float = None):
        self.optimizer     = optimizer
        self.T_max         = n_epochs
        self.initial_lr    = initial_lr
        self.decay_rate    = decay_rate if decay_rate is not None else 10 ** (-3.0 / n_epochs)
        self.current_epoch = 0

    def step(self):
        exp_factor = self.decay_rate ** self.current_epoch
        cos_factor = 0.5 * (1.0 + np.cos(np.pi * self.current_epoch / self.T_max))
        new_lr = self.initial_lr * exp_factor * cos_factor
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr
        self.current_epoch += 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gyro_data(csv_path: str):
    """Load ages, BPRP0, BPRP0_err, Prot, mem_prob, gaia_ids, gaia_ids from phot_all.csv."""
    df = pd.read_csv(csv_path)
    df['GaiaDR3_ID'] = df['GaiaDR3_ID'].astype(str)

    valid_mask = (
        df['age_Myr'].notna() &
        df['BPRP0'].notna() &
        df['BPRP0_err'].notna() &
        df['Prot'].notna() &
        (df['Prot'] > 0)
    )
    df = df[valid_mask].reset_index(drop=True)

    ages      = df['age_Myr'].values.astype(float)
    bprp0     = df['BPRP0'].values.astype(float)
    bprp0_err = df['BPRP0_err'].values.astype(float)
    prot      = df['Prot'].values.astype(float)
    gaia_ids  = df['GaiaDR3_ID'].values

    mem_prob = df['mem_prob_val'].values.astype(float) if 'mem_prob_val' in df.columns \
               else np.full(len(ages), np.nan)

    print(f'Loaded {len(ages)} stars with valid age, BPRP0, BPRP0_err, Prot')
    print(f'Age range:  {ages.min():.1f} – {ages.max():.1f} Myr')
    print(f'Prot range: {prot.min():.2f} – {prot.max():.2f} days')

    return ages, bprp0, bprp0_err, prot, mem_prob, gaia_ids, gaia_ids


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_single_fold(log_prot_train, y_train, bprp0_train, bprp0_err_train, mem_prob_train,
                      log_prot_val,   y_val,   bprp0_val,   bprp0_err_val,   mem_prob_val,
                      lr, weight_decay, n_epochs, batch_size, device,
                      flow_transforms=8, flow_hidden_features=None,
                      loga_grid=None, lr_decay_rate=None):
    """Train one fold NLE model.

    log_prot_*:   (N,) log10(Prot) — the 1D likelihood variable (no PCA needed)
    y_*:          (N,) log10(age/Myr) — flow context
    bprp0_*:      (N,) raw BPRP0 — flow context
    bprp0_err_*:  (N,) log10(BPRP0_err) — flow context
    mem_prob_*:   (N,) cluster membership probability

    Returns:
        val_stats, best_val_nll, best_state, train_losses, val_losses
    """
    loga_grid_np = np.linspace(PRIOR_LOGA_MYR[0], PRIOR_LOGA_MYR[1], LOGA_GRID_DEFAULT_SIZE)
    if loga_grid is None:
        loga_grid = torch.tensor(loga_grid_np, dtype=torch.float32, device=device)

    # log_prot is already 1D; reshape to (N, 1) for the flow
    def to_t(arr):
        return torch.tensor(arr.astype(np.float32), dtype=torch.float32, device=device)

    train_dataset = TensorDataset(
        to_t(log_prot_train).unsqueeze(1),   # (N, 1)
        to_t(y_train),
        to_t(bprp0_train),
        to_t(bprp0_err_train),
        to_t(mem_prob_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model     = GyroNLEPredictor(flow_transforms=flow_transforms,
                                  flow_hidden_features=flow_hidden_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CombinedLRScheduler(optimizer, n_epochs=n_epochs, initial_lr=lr,
                                    decay_rate=lr_decay_rate)

    z_val_t     = to_t(log_prot_val).unsqueeze(1)
    y_val_t     = to_t(y_val)
    bprp0_val_t = to_t(bprp0_val)
    berr_val_t  = to_t(bprp0_err_val)
    mp_val_t    = to_t(mem_prob_val)

    best_loss  = float('inf')
    best_state = None
    train_losses = []
    val_losses   = []

    for _ in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for z_b, y_b, bprp0_b, berr_b, mp_b in train_loader:
            optimizer.zero_grad()
            loss = model(z_b, y_b, bprp0_b, berr_b, mp_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(z_b)
        epoch_loss /= len(log_prot_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_nll = model(z_val_t, y_val_t, bprp0_val_t, berr_val_t, mp_val_t).item()
        val_losses.append(val_nll)
        model.train()

        if val_nll < best_loss:
            best_loss  = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        stats     = model.predict_stats(z_val_t, bprp0_val_t, berr_val_t, loga_grid.to(device))
        val_stats = {k: v.cpu().numpy() for k, v in stats.items()}

    return val_stats, best_loss, best_state, train_losses, val_losses


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------

def run_kfold_cv(ages, bprp0, bprp0_err, prot, mem_prob, gaia_ids,
                 n_folds=5, lr=1e-3, weight_decay=1e-4, n_epochs=300, batch_size=64,
                 device=None, seed=42, save_models_dir=None,
                 flow_transforms=8, flow_hidden_features=None,
                 loga_grid_size=None, lr_decay_rate=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(seed)
    n_samples = len(ages)

    y              = np.log10(ages)
    log_prot       = np.log10(prot)
    bprp0_err_norm = np.log10(np.clip(bprp0_err, 1e-10, None))
    # bprp0 passed raw; log_prot is already on a reasonable scale

    G = loga_grid_size or LOGA_GRID_DEFAULT_SIZE
    loga_grid_t = torch.tensor(np.linspace(PRIOR_LOGA_MYR[0], PRIOR_LOGA_MYR[1], G),
                               dtype=torch.float32)

    # Sample-level fold assignment (all sectors per star already reduced to one row)
    perm           = np.random.permutation(n_samples)
    fold_size      = n_samples // n_folds
    fold_of_sample = np.zeros(n_samples, dtype=int)
    for f in range(n_folds):
        s = f * fold_size
        e = s + fold_size if f < n_folds - 1 else n_samples
        fold_of_sample[perm[s:e]] = f

    stat_keys = ('median', 'mean', 'map', 'p16', 'p84')
    all_stats   = {k: np.zeros(n_samples) for k in stat_keys}
    fold_losses = []
    fold_models = []

    print(f'\nRunning {n_folds}-fold CV on {n_samples} samples...')

    for fold in range(n_folds):
        val_idx   = np.where(fold_of_sample == fold)[0]
        train_idx = np.where(fold_of_sample != fold)[0]

        val_stats, best_nll, model_state, train_losses, val_losses = train_single_fold(
            log_prot[train_idx], y[train_idx], bprp0[train_idx],
            bprp0_err_norm[train_idx], mem_prob[train_idx],
            log_prot[val_idx],   y[val_idx],   bprp0[val_idx],
            bprp0_err_norm[val_idx],   mem_prob[val_idx],
            lr, weight_decay, n_epochs, batch_size, device,
            flow_transforms=flow_transforms,
            flow_hidden_features=flow_hidden_features,
            loga_grid=loga_grid_t,
            lr_decay_rate=lr_decay_rate,
        )

        fold_models.append({
            'state_dict':   model_state,
            'train_losses': train_losses,
            'val_losses':   val_losses,
        })

        for k in stat_keys:
            all_stats[k][val_idx] = val_stats[k]
        fold_losses.append(best_nll)

        val_pred_log = val_stats['median']
        val_mae  = np.mean(np.abs(val_pred_log - y[val_idx]))
        val_corr = np.corrcoef(val_pred_log, y[val_idx])[0, 1]
        print(f'  Fold {fold + 1}/{n_folds}: best_val_nll={best_nll:.4f}  '
              f'val_MAE={val_mae:.3f} dex  val_r={val_corr:.3f}')

    if save_models_dir is not None:
        save_dir = Path(save_models_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'fold_models': fold_models,
            'n_folds':     n_folds,
            'seed':        seed,
        }, save_dir / 'kfold_models.pt')
        print(f'Saved {n_folds} fold models to {save_dir / "kfold_models.pt"}')

        loss_curves = {
            f'fold_{f}': {
                'train_nll': fold_models[f]['train_losses'],
                'val_nll':   fold_models[f]['val_losses'],
            }
            for f in range(n_folds)
        }
        with open(save_dir / 'training_curves.json', 'w') as fp:
            json.dump(loss_curves, fp)

    predictions = all_stats['median']  # log10(age/Myr)
    return predictions, all_stats, ages, bprp0, prot, gaia_ids, fold_of_sample, fold_losses


# ---------------------------------------------------------------------------
# Results / plotting
# ---------------------------------------------------------------------------

def plot_kfold_results(predictions, true_ages, bprp0, prot, gaia_ids,
                       fold_assignments, pred_stats, output_dir):
    log_true  = np.log10(true_ages)
    log_resid = predictions - log_true
    mae       = np.mean(np.abs(log_resid))
    med_ae    = np.median(np.abs(log_resid))
    rmse      = np.sqrt(np.mean(log_resid ** 2))
    bias      = np.mean(log_resid)
    scat      = np.std(log_resid)
    corr      = np.corrcoef(predictions, log_true)[0, 1]

    print(f'\n=== Gyro NLE Baseline Results ===')
    print(f'N = {len(true_ages)}')
    print(f'MAE = {mae:.3f} dex   Median AE = {med_ae:.3f} dex   RMSE = {rmse:.3f} dex')
    print(f'Bias = {bias:.3f} dex   Scatter = {scat:.3f} dex   Pearson r = {corr:.3f}')

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    sc  = ax1.scatter(log_true, predictions, c=bprp0, cmap='coolwarm',
                      alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    lo, hi = log_true.min(), log_true.max()
    ax1.plot([lo, hi], [lo, hi], 'k--', lw=2, label='Perfect')
    ax1.plot([lo, hi], [lo + 0.3, hi + 0.3], 'k:', alpha=0.5, label='±0.3 dex')
    ax1.plot([lo, hi], [lo - 0.3, hi - 0.3], 'k:', alpha=0.5)
    ax1.set_xlabel('log\u2081\u2080(True Age / Myr)'); ax1.set_ylabel('log\u2081\u2080(Predicted Age / Myr)')
    ax1.set_title(f'Gyro NLE baseline (r={corr:.3f})')
    ax1.legend(loc='upper left')
    plt.colorbar(sc, ax=ax1, label='BP-RP\u2080')
    ax1.text(0.02, 0.98,
             f'N={len(true_ages)}\nMAE={mae:.3f} dex\n\u03c3={scat:.3f} dex\nr={corr:.3f}',
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(log_true, log_resid, c=bprp0, cmap='coolwarm',
                alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    ax2.axhline(0, color='k', linestyle='--', lw=2)
    ax2.set_xlabel('log\u2081\u2080(True Age / Myr)'); ax2.set_ylabel('Residual (dex)')
    ax2.set_title('Residuals vs True Age')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(log_resid, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='r', linestyle='--', lw=2, label='Zero')
    ax3.axvline(bias, color='orange', lw=2, label=f'Bias={bias:.3f}')
    ax3.set_xlabel('Residual (dex)'); ax3.set_ylabel('Count')
    ax3.set_title(f'Residuals (\u03c3={scat:.3f} dex)')
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(bprp0, log_resid, c=log_true, cmap='viridis', alpha=0.6, s=15)
    ax4.axhline(0, color='k', linestyle='--', lw=2)
    ax4.set_xlabel('BP-RP\u2080'); ax4.set_ylabel('Residual (dex)')
    ax4.set_title('Residuals vs Colour (by log age)')
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(log_true.min(), log_true.max()))
    plt.colorbar(sm, ax=ax4, label='log\u2081\u2080(Age / Myr)')

    plt.tight_layout()
    plt.savefig(output_dir / 'kfold_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {output_dir / "kfold_results.png"}')

    # Uncertainty intervals
    p16 = pred_stats['p16']
    p84 = pred_stats['p84']

    results_df = pd.DataFrame({
        'gaia_id':         gaia_ids,
        'true_log_age':    log_true,
        'pred_median':     predictions,
        'pred_mean':       pred_stats['mean'],
        'pred_map':        pred_stats['map'],
        'pred_p16':        p16,
        'pred_p84':        p84,
        'residual_dex':    log_resid,
        'bprp0':           bprp0,
        'prot_days':       prot,
        'fold':            fold_assignments,
    })
    results_df.to_csv(output_dir / 'kfold_predictions.csv', index=False)

    metrics = {
        'n_samples':   int(len(true_ages)),
        'mae_dex':     float(mae),
        'bias_dex':    float(bias),
        'scatter_dex': float(scat),
        'correlation': float(corr),
    }
    with open(output_dir / 'kfold_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Gyrochronology NLE baseline: p(log10 Prot | log10_age, BPRP0, log10 BPRP0_err)')

    parser.add_argument('--age_csv',     type=str, default='data/phot_all.csv')
    parser.add_argument('--output_dir',  type=str, default='output/age_predictor_tests/gyro_baseline_nle')
    parser.add_argument('--bprp0_min',   type=float, default=None)
    parser.add_argument('--bprp0_max',   type=float, default=None)

    # Flow architecture
    parser.add_argument('--flow_transforms',  type=int,   default=8)
    parser.add_argument('--flow_hidden_dims', type=int,   nargs='+', default=[64, 64])
    parser.add_argument('--loga_grid_size',   type=int,   default=LOGA_GRID_DEFAULT_SIZE)

    # Training
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--lr_decay_rate', type=float, default=None,
                        help='Exponential LR decay rate per epoch. '
                             'Default: 10^(-3/n_epochs).')
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--n_epochs',      type=int,   default=300)
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--n_folds',       type=int,   default=5)
    parser.add_argument('--seed',          type=int,   default=42)

    parser.add_argument('--load_latents', type=str, default=None,
                        help='Path to latents cache (.npz). If provided, gyro data is '
                             'filtered to only stars present in the cache (i.e. stars '
                             'with TESS light curves).')

    args   = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('\n=== Loading gyro data ===')
    ages, bprp0, bprp0_err, prot, mem_prob, gaia_ids, _ = load_gyro_data(args.age_csv)

    if args.load_latents:
        print(f'\n=== Filtering to stars with light curves (from {args.load_latents}) ===')
        cache = np.load(args.load_latents, allow_pickle=True)
        lc_gaia_ids = set(cache['gaia_ids'].astype(str))
        mask = np.array([g in lc_gaia_ids for g in gaia_ids])
        n_before = len(ages)
        ages, bprp0, bprp0_err, prot, mem_prob, gaia_ids = (
            a[mask] for a in (ages, bprp0, bprp0_err, prot, mem_prob, gaia_ids))
        print(f'  {n_before} → {len(ages)} stars (kept {mask.sum()} with light curves)')

    if args.bprp0_min is not None:
        mask = bprp0 >= args.bprp0_min
        ages, bprp0, bprp0_err, prot, mem_prob, gaia_ids = (
            a[mask] for a in (ages, bprp0, bprp0_err, prot, mem_prob, gaia_ids))
    if args.bprp0_max is not None:
        mask = bprp0 <= args.bprp0_max
        ages, bprp0, bprp0_err, prot, mem_prob, gaia_ids = (
            a[mask] for a in (ages, bprp0, bprp0_err, prot, mem_prob, gaia_ids))

    print('\n=== Running K-Fold CV ===')
    predictions, pred_stats, true_ages, bprp0, prot, gaia_ids, fold_assignments, fold_losses = run_kfold_cv(
        ages, bprp0, bprp0_err, prot, mem_prob, gaia_ids,
        n_folds=args.n_folds,
        lr=args.lr,
        lr_decay_rate=args.lr_decay_rate,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        save_models_dir=output_dir,
        flow_transforms=args.flow_transforms,
        flow_hidden_features=args.flow_hidden_dims,
        loga_grid_size=args.loga_grid_size,
    )

    metrics = plot_kfold_results(
        predictions, true_ages, bprp0, prot, gaia_ids,
        fold_assignments, pred_stats, output_dir,
    )

    config = vars(args)
    config.update({'n_samples': int(len(true_ages)), 'fold_losses': fold_losses})
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print('\nDone!')


if __name__ == '__main__':
    main()
