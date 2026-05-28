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
import kfold_age_inference as _kai  # noqa: E402
from kfold_age_inference import (  # noqa: E402
    run_kfold_cv,
    fit_global_pca_bundle,
    save_global_pca_artifact,
    load_global_pca_artifact,
)

# Hosts have no cluster-membership concept — each host carries its own literature
# age, not a probabilistic cluster assignment. Disable the static-outlier mixture
# entirely so training NLL is the pure flow log-prob. With P_OUTLIER=0 and
# mem_prob=1.0 (forced below) the flow weight is nf_weight = 1·(1−0) = 1, and the
# outlier term zeros out under logsumexp. Inference (predict_stats) was already
# mixture-free, so this only affects the training loss.
_kai.P_OUTLIER = 0.0


def plot_results_with_uncertainty(true_age, stats, output_dir: Path, axis_label: str):
    valid = ~np.isnan(stats['median'])
    t   = true_age[valid]
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
    ax.set_xlabel(f'True {axis_label}')
    ax.set_ylabel(f'Predicted {axis_label} — median ± [p16, p84]')
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
    parser.add_argument('--host_age_col', type=str, default='st_age',
                        help="CSV column carrying the central host age. Units must match "
                             "--age_space: 'log10_myr'/'gyr' expect Gyr (e.g. st_age); "
                             "'passthrough' takes the column as-is (e.g. st_age_norm).")
    parser.add_argument('--host_age_err_col', type=str, default=None,
                        help="CSV column carrying the per-star age 1σ in the SAME units as "
                             "--host_age_col (Gyr for st_age, normalized for st_age_norm). "
                             "Required when --k_age_samples > 1.")
    parser.add_argument('--age_space', type=str, default='log10_myr',
                        choices=['log10_myr', 'gyr', 'passthrough'],
                        help="Target space the flow learns over. 'log10_myr': sample in Gyr, "
                             "clip ≥1 Myr, log10(*1000) per sample → flow learns log10(age/Myr). "
                             "'gyr': sample in Gyr, flow learns age in Gyr directly. "
                             "'passthrough': feed --host_age_col straight to the flow (used "
                             "for already-normalized columns like st_age_norm).")
    parser.add_argument('--k_age_samples', type=int, default=1,
                        help="Per-star Gaussian age samples drawn from N(age, age_err) in the "
                             "CSV units, then transformed per --age_space. Loss is averaged "
                             "per-star over K, then across the batch. Default 1 disables sampling.")
    parser.add_argument('--eiv', action='store_true',
                        help="Errors-in-variables (LatentNN-style): treat each star's "
                             "literature age as a noisy observation of a learnable latent. "
                             "Adds the regularizer 0.5·((y_obs−y_latent)/σ)² to the flow NLL; "
                             "y_latent is updated jointly with the flow and discarded after "
                             "training. Mutually exclusive with --k_age_samples>1; requires "
                             "--host_age_err_col. Currently PCA + joint training only.")
    parser.add_argument('--eiv_sigma_floor_frac', type=float, default=0.05,
                        help="Floor applied to per-star σ as a fraction of dataset std(y). "
                             "Stops stars with NaN/0 literature error from blowing up the "
                             "regularizer (default 5%% — clean labels stay strongly anchored).")
    parser.add_argument('--host_metadata_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--star_aggregation', type=str, default='latent_max',
                        choices=['latent_mean', 'latent_median', 'latent_max', 'latent_mean_std'])
    parser.add_argument('--use_mg', action='store_true',
                        help='Include log10(MG_quick) as 4th flow context variable.')
    parser.add_argument('--prediction_mode', type=str, default='nle',
                        choices=['nle', 'npe'],
                        help="'nle': flow models p(z | age, colours); posterior via Bayes "
                             "on a likelihood grid. 'npe': flow models p(age | z, colours) "
                             "directly — the bottleneck (learned MLP/linear, or the fixed "
                             "PCA projection) enters as flow context, age is the 1D output.")
    parser.add_argument('--npe_standardize_target', action=argparse.BooleanOptionalAction,
                        default=True,
                        help="NPE only: z-score the age target by one GLOBAL (loc, scale) "
                             "before the flow (undone at predict time, so predictions stay in "
                             "input units). The NSF spline has a hard ±5 support, so raw Gyr "
                             "ages saturate at ~5 Gyr without this. Shared across folds + the "
                             "full model; baked into model buffers. No-op for NLE. Disable "
                             "with --no-npe_standardize_target (e.g. age_space=log10_myr, "
                             "whose 0–4.14 range already fits the spline).")

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

    # Global PCA cache (PCA encoder only): reuses the same basis the pretrain
    # PCA4 workflow uses (final_model/sendit/e50/age_inference/shared/global_pca_d16.npz).
    # If --pca_cache exists, load + truncate to --bottleneck_dim. Otherwise build
    # from --pca_latent_pool at --pca_cache_max_dim and save.
    parser.add_argument('--pca_cache', type=str, default=None, metavar='PATH',
                        help='Global PCA artifact path (load if exists, else build from '
                             '--pca_latent_pool at --pca_cache_max_dim and save).')
    parser.add_argument('--pca_cache_max_dim', type=int, default=16,
                        help='Components to fit when CREATING --pca_cache. Loaded runs '
                             'truncate to --bottleneck_dim.')
    parser.add_argument('--pca_latent_pool', type=str, nargs='+', default=None, metavar='PATH',
                        help='One or more latents npz files concatenated to fit the global '
                             'PCA basis (only used when --pca_cache is being created).')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1. Load + aggregate
    latents, ages, bprp0, bprp0_err, mg, gaia_ids = load_host_latents(
        args.load_latents, args.host_age_csv, args.host_metadata_csv,
        age_col=args.host_age_col)

    # load_host_latents pre-multiplies st_age by 1000 (Gyr → Myr) so the legacy
    # log10-Myr pipeline could do np.log10(ages) directly. Our new transform
    # path (to_target_space / sigma_to_target_space) expects CSV-native units
    # — Gyr for st_age, normalized for st_age_norm — so undo that scale here,
    # giving us a single unambiguous unit assumption per --age_space. σ from
    # st_ageerr is already CSV-native (the loader never touched it).
    if args.host_age_col == 'st_age':
        ages = ages / 1000.0   # Myr → Gyr

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

    # Outlier mixture disabled at module level (P_OUTLIER=0 above). Force
    # mem_prob=1 per star so nf_weight = 1·(1−0) = 1 and the flow log-prob is
    # the entire training NLL.
    mem_prob = np.ones(len(star_age), dtype=np.float32)

    # Per-star age uncertainty: looked up from the age CSV by GaiaDR3_ID and
    # aligned to the post-aggregation star order. Either K-Gaussian-sampling
    # (k_age_samples>1) or EIV needs this; both flags are mutually exclusive.
    if args.eiv and args.k_age_samples > 1:
        raise ValueError('--eiv is mutually exclusive with --k_age_samples > 1 '
                         '(both target attenuation bias from noisy targets — pick one).')
    need_sigma = args.k_age_samples > 1 or args.eiv
    if need_sigma:
        if args.host_age_err_col is None:
            raise ValueError('--k_age_samples > 1 and --eiv both require --host_age_err_col.')
        df_err = pd.read_csv(args.host_age_csv)
        df_err['GaiaDR3_ID'] = df_err['GaiaDR3_ID'].astype(str)
        if args.host_age_err_col not in df_err.columns:
            raise KeyError(f"{args.host_age_err_col!r} not in {args.host_age_csv}")
        err_map = dict(zip(df_err['GaiaDR3_ID'], df_err[args.host_age_err_col]))
        star_age_err = np.array(
            [err_map.get(g, np.nan) for g in star_gids], dtype=np.float32)
        n_with_err = int(np.sum(~np.isnan(star_age_err) & (star_age_err > 0)))
        print(f'Loaded per-star σ from {args.host_age_err_col}: '
              f'{n_with_err}/{len(star_age_err)} stars have non-zero uncertainty.')
    else:
        star_age_err = None

    # ── Age-space transform + K-Gaussian sampling (done in CSV units, then
    # transformed per-sample). Result is a (N, K) target array fed straight to
    # the flow with skip_log10=True; run_kfold_cv's internal log10 + K-sampling
    # paths are bypassed so units stay coherent end-to-end.
    GRID_RANGES = {
        'log10_myr':   (0.0, 4.14),    # 1 Myr → ~14 Gyr (matches PRIOR_LOGA_MYR)
        'gyr':         (0.0, 14.0),
        'passthrough': (-2.5, 5.0),    # covers st_age_norm range
    }
    AGE_LABELS = {
        'log10_myr':   'log10(age / Myr)',
        'gyr':         'age / Gyr',
        'passthrough': f'age ({args.host_age_col})',
    }

    def to_target_space(vals: np.ndarray) -> np.ndarray:
        """Transform CSV-unit ages to the flow's target space (see --age_space)."""
        if args.age_space == 'log10_myr':
            # vals assumed Gyr; clip ≥1 Myr to keep within the grid
            return np.log10(np.clip(vals, 1e-3, None) * 1000.0)
        if args.age_space == 'gyr':
            return np.clip(vals, 1e-3, None)
        return vals  # passthrough

    def sigma_to_target_space(age_csv: np.ndarray, sigma_csv: np.ndarray) -> np.ndarray:
        """Convert per-star σ from CSV units to flow target-space units.

        log10_myr: delta-method σ_log10 ≈ σ_Gyr / (age · ln10) with age clipped
                   at 1 Myr — matches the central-value transform's clip.
        gyr / passthrough: identity (σ already in target units).
        """
        if args.age_space == 'log10_myr':
            age_clipped = np.clip(age_csv, 1e-3, None)
            return (sigma_csv / (age_clipped * np.log(10.0))).astype(np.float32)
        return sigma_csv.astype(np.float32)

    # Build (N, K) targets in CSV units, then transform per-sample.
    K = args.k_age_samples
    if K > 1:
        rng = np.random.default_rng(args.seed)
        sigma = np.where(np.isnan(star_age_err), 0.0,
                         np.maximum(star_age_err, 0.0)).astype(np.float32)
        noise = rng.standard_normal(size=(len(star_age), K)).astype(np.float32)
        raw_targets = (star_age[:, None] + sigma[:, None] * noise).astype(np.float32)
        print(f'Drew {K} Gaussian age samples per star in CSV units '
              f'({int(np.sum(sigma > 0))}/{len(sigma)} have non-zero σ; '
              f'rest get K identical copies).')
        y_targets = to_target_space(raw_targets)            # (N, K) on target scale
    else:
        y_targets = to_target_space(star_age)               # (N,) on target scale

    # Per-star central value on the target scale (for printable metrics +
    # plot/CSV true axis — independent of any K-sample noise).
    y_central = to_target_space(star_age)
    age_grid_range = GRID_RANGES[args.age_space]
    age_label      = AGE_LABELS[args.age_space]
    print(f'Age space: {args.age_space}  |  grid range: '
          f'[{age_grid_range[0]:.2f}, {age_grid_range[1]:.2f}]  |  '
          f'target shape: {y_targets.shape}')

    # 2a. Global PCA cache (PCA encoder only). Mirrors kfold_age_inference.py
    # step 2d: load if the file exists, otherwise build from --pca_latent_pool
    # and save. The cache supersedes any per-fold PCA refit inside run_kfold_cv.
    prefit_pca_bundle = None
    if args.pca_cache:
        import os
        if os.path.exists(args.pca_cache):
            print(f'\n=== Loading global PCA cache: {args.pca_cache} ===')
            pca_obj_c, X_mean_c, X_std_c = load_global_pca_artifact(
                args.pca_cache, args.bottleneck_dim)
            prefit_pca_bundle = (pca_obj_c, X_mean_c, X_std_c)
        else:
            if not args.pca_latent_pool:
                parser.error('--pca_cache file does not exist and --pca_latent_pool was not '
                             'provided; cannot compute the cache.')
            print(f'\n=== Loading PCA fit pool ({len(args.pca_latent_pool)} cache(s)) ===')
            pools = []
            for p in args.pca_latent_pool:
                with np.load(p, allow_pickle=True) as data:
                    pools.append(np.asarray(data['latent_vectors']))
                    print(f'  {p}: {pools[-1].shape}')
            pca_pool = np.concatenate(pools, axis=0)
            del pools
            print(f'Global PCA pool: {len(pca_pool):,} latents x {pca_pool.shape[1]} dims')
            print(f'\n=== Computing global PCA cache (max_dim={args.pca_cache_max_dim}) ===')
            pca_full, Xm, Xs = fit_global_pca_bundle(pca_pool, args.pca_cache_max_dim)
            save_global_pca_artifact(args.pca_cache, pca_full, Xm, Xs)
            del pca_full, pca_pool
            pca_obj_c, X_mean_c, X_std_c = load_global_pca_artifact(
                args.pca_cache, args.bottleneck_dim)
            prefit_pca_bundle = (pca_obj_c, X_mean_c, X_std_c)

    # σ in TARGET units for EIV (run_kfold_cv expects age_err in target space).
    # When EIV is off, we still pass None so the upstream sampling path stays
    # untouched (K-sampling was done in to_target_space above; age_err=None is
    # the right signal there).
    age_err_target = (sigma_to_target_space(star_age, star_age_err)
                      if args.eiv and star_age_err is not None else None)

    # 2b. K-fold NLE. Targets (y_targets) are already on the flow's target scale
    # — pass skip_log10=True and k_age_samples=1 so run_kfold_cv's internal
    # transform + sampling paths stay out of the way. When K>1, y_targets is
    # (N, K) and run_kfold_cv slices it row-wise per fold; the model forward
    # already handles log_age.dim() == 2 (per-star K-sample mean NLL).
    predictions, all_stats, _, _, fold_assignments, fold_losses, _ = run_kfold_cv(
        latent_vectors=star_lat.astype(np.float32),
        ages=y_targets.astype(np.float32),    # pre-transformed; 1D if K=1, 2D if K>1
        bprp0=star_b.astype(np.float32),
        bprp0_err=star_be.astype(np.float32),
        mg=star_mg.astype(np.float32),
        mem_prob=mem_prob,
        tic_ids=star_gids,                    # gaia ids stand in for tic_ids
        n_folds=args.n_folds,
        pca_dim=args.bottleneck_dim,
        pca_latents=None,                     # superseded by prefit_pca_bundle when set
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
        skip_log10=True,                      # all transforms done above
        age_grid_range=age_grid_range,
        age_err=age_err_target,               # target-space σ; only used when eiv=True
        k_age_samples=1,                      # K-sampling done above (mutually excl. with EIV)
        prediction_mode=args.prediction_mode,
        prefit_pca_bundle=prefit_pca_bundle,
        eiv=args.eiv,
        eiv_sigma_floor_frac=args.eiv_sigma_floor_frac,
        # NPE models age directly through the ±5-bounded NSF spline, so raw Gyr
        # targets (0–15) saturate at ~5 Gyr. Standardize the target for NPE (one
        # global loc/scale shared across all folds; predictions stay in input
        # units). Gated to npe+pca inside run_kfold_cv; no-op for NLE.
        npe_standardize_target=args.npe_standardize_target,
    )

    # 3. Plots + predictions. The "true" axis is the transformed central age;
    # all predictions are already on the target scale.
    overall = plot_results_with_uncertainty(y_central, all_stats, output_dir, age_label)

    # Predictions CSV: a common block (true central + posterior summary on the
    # target scale) plus an age-space-specific block (linear-Myr columns when
    # we're learning log10_myr, raw CSV column when passthrough, Gyr otherwise).
    common = {
        'GaiaDR3_ID':       star_gids,
        f'true_{args.age_space}': y_central,
        f'pred_{args.age_space}_median': all_stats['median'],
        f'pred_{args.age_space}_p16':    all_stats['p16'],
        f'pred_{args.age_space}_p84':    all_stats['p84'],
        f'pred_{args.age_space}_mean':   all_stats['mean'],
        f'pred_{args.age_space}_map':    all_stats['map'],
        'BPRP0':            star_b,
        'BPRP0_err':        star_be,
        'fold':             fold_assignments,
    }
    if args.age_space == 'log10_myr':
        # tag on linear-Myr columns for convenience
        extra = {
            'true_age_Myr':        10 ** y_central,
            'pred_age_median_Myr': 10 ** all_stats['median'],
            'pred_age_p16_Myr':    10 ** all_stats['p16'],
            'pred_age_p84_Myr':    10 ** all_stats['p84'],
        }
        df_out = pd.DataFrame({**common, **extra})
    else:
        df_out = pd.DataFrame(common)

    # Tag on raw archive ages for reference (whichever of these columns exist
    # in the age CSV — st_age / st_ageerr live in the default file, the _norm
    # variants in the normalized file, both for the merged file).
    archive_cols = ['st_age', 'st_ageerr', 'st_age_norm', 'st_ageerr_norm']
    df_arch = pd.read_csv(args.host_age_csv)
    df_arch['GaiaDR3_ID'] = df_arch['GaiaDR3_ID'].astype(str)
    keep = ['GaiaDR3_ID'] + [c for c in archive_cols if c in df_arch.columns]
    if len(keep) > 1:
        df_out = df_out.merge(df_arch[keep].drop_duplicates('GaiaDR3_ID'),
                              on='GaiaDR3_ID', how='left')

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
    # Record the global NPE target standardization actually used (deterministic
    # from y_central — identical to what run_kfold_cv computed and baked into the
    # model buffers). Lets a later run / loader confirm the same normalization.
    if args.prediction_mode == 'npe':
        summary['npe_target_standardization'] = {
            'loc':   float(np.mean(y_central)),
            'scale': float(np.std(y_central)),
            'units': args.age_space,
            'note':  'global over all labeled stars; shared across folds + full model',
        }
    with (output_dir / 'metrics.json').open('w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'Saved {output_dir/"metrics.json"}')

    print('\nDone.')
    print(f'Overall: MAE={overall["mae"]:.4f}  RMSE={overall["rmse"]:.4f}  '
          f'r={overall["pearson_r"]:.3f}  n={overall["n"]}')


if __name__ == '__main__':
    main()
