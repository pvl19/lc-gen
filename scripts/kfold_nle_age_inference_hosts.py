"""K-fold normalizing-flow age inference for exoplanet hosts.

Conceptually parallel to scripts/kfold_age_inference.py but with host-specific data
loading (cached host latents + archive_ages CSV by GaiaDR3_ID, instead of the
pretrain H5 + all_ages.csv path). The model architecture and 3-stage training
schedule are unchanged: an MLP encoder compresses latents to a bottleneck, an
NSF flow models p(z | log10_age, BPRP0, log10(BPRP0_err), [log10(MG)]),
and posterior stats are extracted from a likelihood grid.

Per-star outputs include the posterior median, p16, p84 (plus mean and MAP).

Pipeline:
    1. Load cached host latents (kfold-format or predict_ages format), join ages
       and BPRP0/MG metadata by GaiaDR3_ID.
    2. Per-sector → per-star aggregation (latent_mean / latent_median /
       latent_max / latent_mean_std).
    3. Stratified k-fold over stars; per fold: 3-stage MLP encoder + NSF flow.
    4. Aggregate predictions (median, p16, p84, mean, MAP), write CSV +
       metrics + scatter with error bars.
    5. Optionally train one deployment model on all stars (--train_full).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Allow importing sibling scripts (kfold_mlp_age_inference, kfold_age_inference)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from kfold_mlp_age_inference import load_host_latents, aggregate_by_star  # noqa: E402
from kfold_age_inference import run_kfold_cv  # noqa: E402


def plot_results_with_uncertainty(true_log, stats, output_dir: Path):
    valid = ~np.isnan(stats['median'])
    t   = true_log[valid]
    med = stats['median'][valid]
    p16 = stats['p16'][valid]
    p84 = stats['p84'][valid]
    err_lo = np.clip(med - p16, 0, None)
    err_hi = np.clip(p84 - med, 0, None)

    mae  = float(np.mean(np.abs(med - t)))
    rmse = float(np.sqrt(np.mean((med - t) ** 2)))
    r    = float(pearsonr(med, t)[0]) if len(t) > 2 else float('nan')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.errorbar(t, med, yerr=[err_lo, err_hi], fmt='o', ms=4, alpha=0.5,
                ecolor='gray', elinewidth=0.7, capsize=0)
    lo, hi = float(min(t.min(), med.min())), float(max(t.max(), med.max()))
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, lw=1)
    ax.set_xlabel('True log10(age / Myr)')
    ax.set_ylabel('Predicted log10(age / Myr) — median ± [p16, p84]')
    ax.set_title('K-fold NLE age inference (hosts)\n'
                 f'MAE={mae:.3f}  RMSE={rmse:.3f}  r={r:.3f}  n={len(t)}')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = output_dir / 'scatter_pred_vs_true.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
    return {'mae': mae, 'rmse': rmse, 'pearson_r': r, 'n': int(len(t))}


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
    parser.add_argument('--use_mg', action='store_true',
                        help='Include log10(MG_quick) as 4th flow context variable.')

    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    # Encoder
    parser.add_argument('--encoder_type', type=str, default='mlp',
                        choices=['pca', 'mlp', 'linear'])
    parser.add_argument('--bottleneck_dim', type=int, default=4,
                        help='PCA dim or MLP bottleneck dim (passed to run_kfold_cv as pca_dim).')
    parser.add_argument('--mlp_encoder_hidden', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--aux_loss_weight', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--variance_reg_weight', type=float, default=0.25)

    # Training stages
    parser.add_argument('--training_stages', type=str, default='three_stage',
                        choices=['joint', 'two_stage', 'three_stage'])
    parser.add_argument('--encoder_pretrain_epochs', type=int, default=100)
    parser.add_argument('--joint_finetune_epochs', type=int, default=100)
    parser.add_argument('--finetune_encoder_lr_mult', type=float, default=0.001)
    parser.add_argument('--finetune_flow_lr_mult', type=float, default=0.1)

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_rate', type=float, default=0.97)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)

    # Flow
    parser.add_argument('--flow_transforms', type=int, default=6)
    parser.add_argument('--flow_hidden_dims', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--loga_grid_size', type=int, default=1000)

    parser.add_argument('--train_full', action='store_true',
                        help='After k-fold, train one deployment model on all stars.')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1. Load + aggregate
    latents, ages, bprp0, bprp0_err, mg, gaia_ids = load_host_latents(
        args.load_latents, args.host_age_csv, args.host_metadata_csv)

    valid = ~np.isnan(ages) & ~np.isnan(bprp0) & ~np.isnan(bprp0_err)
    if args.use_mg:
        valid &= ~np.isnan(mg)
    n_drop = int(np.sum(~valid))
    if n_drop > 0:
        print(f'Dropping {n_drop}/{len(ages)} rows with NaN in required fields')
    latents, ages, bprp0, bprp0_err, mg, gaia_ids = (
        latents[valid], ages[valid], bprp0[valid],
        bprp0_err[valid], mg[valid], gaia_ids[valid])

    star_lat, star_age, star_b, star_be, star_mg, star_gids = aggregate_by_star(
        latents, ages, bprp0, bprp0_err, mg, gaia_ids, method=args.star_aggregation)
    print(f'Final dataset: {len(star_lat)} stars | latent dim: {star_lat.shape[1]}')

    # Hosts have no cluster-membership probability — pass NaN so the outlier
    # model in AgePredictorMLP._nll_with_outlier falls back to the constant
    # P_CLUSTER_MEM (0.9). Same convention as predict_ages.
    mem_prob = np.full(len(star_age), np.nan, dtype=np.float32)

    # 2. K-fold NLE
    predictions, all_stats, true_ages, _, fold_assignments, fold_losses, _ = run_kfold_cv(
        latent_vectors=star_lat.astype(np.float32),
        ages=star_age.astype(np.float32),
        bprp0=star_b.astype(np.float32),
        bprp0_err=star_be.astype(np.float32),
        mg=star_mg.astype(np.float32),
        mem_prob=mem_prob,
        tic_ids=star_gids,                    # gaia ids stand in for tic_ids
        n_folds=args.n_folds,
        pca_dim=args.bottleneck_dim,
        pca_latents=None,                     # no global PCA prefit
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        save_models_dir=str(output_dir),
        star_level_split=False,               # already aggregated per star
        flow_transforms=args.flow_transforms,
        flow_hidden_features=args.flow_hidden_dims,
        loga_grid_size=args.loga_grid_size,
        encoder_type=args.encoder_type,
        mlp_encoder_hidden=args.mlp_encoder_hidden,
        aux_loss_weight=args.aux_loss_weight,
        use_mg=args.use_mg,
        lr_decay_rate=args.lr_decay_rate,
        dropout=args.dropout,
        variance_reg_weight=args.variance_reg_weight,
        training_stages=args.training_stages,
        encoder_pretrain_epochs=args.encoder_pretrain_epochs,
        joint_finetune_epochs=args.joint_finetune_epochs,
        finetune_encoder_lr_mult=args.finetune_encoder_lr_mult,
        finetune_flow_lr_mult=args.finetune_flow_lr_mult,
        train_full=args.train_full,
    )

    # 3. Plots + predictions
    log_ages_true = np.log10(true_ages)
    overall = plot_results_with_uncertainty(log_ages_true, all_stats, output_dir)

    df_out = pd.DataFrame({
        'GaiaDR3_ID': star_gids,
        'true_log10_age_Myr':    log_ages_true,
        'true_age_Myr':          true_ages,
        'pred_log10_age_median': all_stats['median'],
        'pred_log10_age_p16':    all_stats['p16'],
        'pred_log10_age_p84':    all_stats['p84'],
        'pred_log10_age_mean':   all_stats['mean'],
        'pred_log10_age_map':    all_stats['map'],
        'pred_age_median_Myr':   10 ** all_stats['median'],
        'pred_age_p16_Myr':      10 ** all_stats['p16'],
        'pred_age_p84_Myr':      10 ** all_stats['p84'],
        'BPRP0':                 star_b,
        'BPRP0_err':             star_be,
        'fold':                  fold_assignments,
    })
    pred_path = output_dir / 'predictions.csv'
    df_out.to_csv(pred_path, index=False)
    print(f'Saved {pred_path}')

    summary = {
        'overall':    overall,
        'config':     vars(args),
        'feature_dim': int(star_lat.shape[1]),
        'n_stars':    int(len(star_lat)),
        'fold_best_val_nll': [float(x) for x in fold_losses],
    }
    with (output_dir / 'metrics.json').open('w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'Saved {output_dir/"metrics.json"}')

    print('\nDone.')
    print(f'Overall: MAE={overall["mae"]:.4f}  RMSE={overall["rmse"]:.4f}  '
          f'r={overall["pearson_r"]:.3f}  n={overall["n"]}')


if __name__ == '__main__':
    main()
