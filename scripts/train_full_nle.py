"""Train a single NLE model on ALL labeled data (no cross-validation).

Use this to produce a model for inference on new unlabeled stars.
Loads latents from a pre-computed cache to avoid re-running the autoencoder.

Usage:
    python scripts/train_full_nle.py \
        --load_latents output/latents_cache/cf_final_multiscale.npz \
        --age_csv      data/phot_all.csv \
        --output_dir   output/full_nle
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from kfold_age_inference import (
    load_latents_cache, aggregate_by_star, train_single_fold,
    PRIOR_LOGA_MYR, LOGA_GRID_DEFAULT_SIZE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_latents',    required=True,
                        help='Path to latents cache (.npz)')
    parser.add_argument('--age_csv',         default='data/phot_all.csv')
    parser.add_argument('--output_dir',      default='output/full_nle')
    parser.add_argument('--star_aggregation',default='latent_max',
                        help='How to aggregate per-sector latents per star')
    parser.add_argument('--pca_dim',         type=int,   default=4)
    parser.add_argument('--val_frac',        type=float, default=0.1,
                        help='Fraction of stars held out for early-stopping')
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--lr_decay_rate',   type=float, default=0.97)
    parser.add_argument('--weight_decay',    type=float, default=1e-4)
    parser.add_argument('--n_epochs',        type=int,   default=100)
    parser.add_argument('--batch_size',      type=int,   default=64)
    parser.add_argument('--flow_transforms', type=int,   default=12)
    parser.add_argument('--flow_hidden_dims',type=int, nargs='+', default=[64, 164])
    parser.add_argument('--loga_grid_size',  type=int,   default=LOGA_GRID_DEFAULT_SIZE)
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--use_mg',          action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load latents cache ──────────────────────────────────────────────────
    print(f'\n=== Loading latents: {args.load_latents} ===')
    (latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors,
     _, _) = load_latents_cache(args.load_latents)

    # Load BPRP0_err, MG, mem_prob from CSV (not stored in cache)
    _df = pd.read_csv(args.age_csv)
    _df['GaiaDR3_ID'] = _df['GaiaDR3_ID'].astype(str)
    id_to_bprp0_err = dict(zip(_df['GaiaDR3_ID'], _df['BPRP0_err']))
    id_to_mg        = dict(zip(_df['GaiaDR3_ID'], _df['MG_quick']))      if 'MG_quick'      in _df.columns else {}
    id_to_mem_prob  = dict(zip(_df['GaiaDR3_ID'], _df['mem_prob_val']))   if 'mem_prob_val'  in _df.columns else {}
    bprp0_err = np.array([id_to_bprp0_err.get(g, np.nan) for g in gaia_ids], dtype=float)
    mg        = np.array([id_to_mg.get(g, np.nan)        for g in gaia_ids], dtype=float)
    mem_prob  = np.array([id_to_mem_prob.get(g, np.nan)  for g in gaia_ids], dtype=float)

    # ── Aggregate per-sector latents to one per star ────────────────────────
    print(f'\n=== Aggregating by star ({args.star_aggregation}) ===')
    latents, ages, bprp0, bprp0_err, mg, mem_prob, star_gaia_ids = aggregate_by_star(
        latent_vectors, ages, bprp0, bprp0_err, mg, mem_prob,
        tic_ids, gaia_ids, method=args.star_aggregation,
    )
    print(f'  {len(ages)} unique stars')

    # ── Normalise latents ───────────────────────────────────────────────────
    norm_mean = latents.mean(axis=0)
    norm_std  = latents.std(axis=0)
    norm_params = {'mean': norm_mean, 'std': norm_std}
    X_norm = (latents - norm_mean) / (norm_std + 1e-8)

    y             = np.log10(ages)
    log_bprp0_err = np.log10(np.clip(bprp0_err, 1e-10, None))
    log_mg        = np.log10(np.clip(mg, 1e-10, None)) if args.use_mg else np.zeros(len(ages))

    # ── Train / val split ───────────────────────────────────────────────────
    n         = len(ages)
    perm      = np.random.permutation(n)
    n_val     = max(1, int(n * args.val_frac))
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]
    print(f'  Train: {len(train_idx)}   Val: {len(val_idx)}')

    loga_grid = torch.tensor(
        np.linspace(PRIOR_LOGA_MYR[0], PRIOR_LOGA_MYR[1], args.loga_grid_size),
        dtype=torch.float32,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print('\n=== Training ===')
    val_stats, best_nll, best_state, pca, train_losses, val_losses = train_single_fold(
        X_norm[train_idx], y[train_idx],       bprp0[train_idx],
        log_bprp0_err[train_idx],              log_mg[train_idx],  mem_prob[train_idx],
        X_norm[val_idx],   y[val_idx],         bprp0[val_idx],
        log_bprp0_err[val_idx],                log_mg[val_idx],    mem_prob[val_idx],
        pca_dim=args.pca_dim,
        lr=args.lr, weight_decay=args.weight_decay,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        device=device,
        flow_transforms=args.flow_transforms,
        flow_hidden_features=args.flow_hidden_dims,
        loga_grid=loga_grid,
        encoder_type='pca',
        use_mg=args.use_mg,
        lr_decay_rate=args.lr_decay_rate,
    )

    val_pred = val_stats['median']
    val_mae  = np.mean(np.abs(val_pred - y[val_idx]))
    val_corr = np.corrcoef(val_pred, y[val_idx])[0, 1]
    print(f'  best_val_nll={best_nll:.4f}  val_MAE={val_mae:.3f} dex  val_r={val_corr:.3f}')

    # ── Save ─────────────────────────────────────────────────────────────────
    model_path = output_dir / 'full_nle_model.pt'
    torch.save({
        'state_dict':           best_state,
        'pca':                  pca,
        'normalization_params': norm_params,
        'pca_dim':              args.pca_dim,
        'use_mg':               args.use_mg,
        'flow_transforms':      args.flow_transforms,
        'flow_hidden_dims':     args.flow_hidden_dims,
        'train_losses':         train_losses,
        'val_losses':           val_losses,
    }, model_path)
    print(f'Saved → {model_path}')

    with open(output_dir / 'training_curves.json', 'w') as fp:
        json.dump({'train_nll': train_losses, 'val_nll': val_losses}, fp)

    config = vars(args)
    config.update({'n_train': int(len(train_idx)), 'n_val': int(len(val_idx))})
    with open(output_dir / 'config.json', 'w') as fp:
        json.dump(config, fp, indent=2, default=str)

    print('Done.')


if __name__ == '__main__':
    main()
