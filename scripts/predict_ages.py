"""Infer ages for unlabeled stars (e.g. exoplanet hosts) using a trained k-fold NLE model.

Loads a model checkpoint produced by kfold_age_inference.py (auto-detects
full_model.pt or kfold_models.pt), applies the same normalisation and encoding
as during training, and runs posterior inference. When using kfold_models.pt,
posteriors are averaged across K folds before extracting summary statistics.

Usage (first time — extract latents from autoencoder):
    python predict_ages.py \\
        --nle_dir    final_model/final_parallel_e10/nf-new-mlp4-multiscale-latent_max-mlp \\
        --ae_model   final_model/final_parallel_e10/model.pt \\
        --h5_path    exop_hosts/lc_data.h5 \\
        --output_dir exop_hosts/exop_host_ages-v2 \\
        --save_latents exop_hosts/exop_host_ages-v2/latents.npz

Usage (subsequent runs — skip autoencoder):
    python predict_ages.py \\
        --nle_dir      final_model/final_parallel_e10/nf-new-mlp4-multiscale-latent_max-mlp \\
        --load_latents exop_hosts/exop_host_ages-v1/latents.npz \\
        --output_dir   exop_hosts/exop_host_ages-v2
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / 'src'))
sys.path.insert(0, str(_repo_root / 'scripts'))
from kfold_age_inference import (
    AgePredictor, AgePredictorMLP, PRIOR_LOGA_MYR, LOGA_GRID_DEFAULT_SIZE,
)
from lcgen.models.TimeSeriesDataset import METADATA_FEATURES
from lcgen.models.MetadataAgePredictor import MetadataStandardizer


# ---------------------------------------------------------------------------
# Latent extraction (reused from scripts/predict_ages.py)
# ---------------------------------------------------------------------------

def load_and_filter_h5_metadata(h5_path: str):
    """Read metadata from H5 and apply strict filtering (no NaN substitution).

    Also reads MG_quick / MG_quick_err if present (added by add_mg_quick.py); falls
    back to NaN-filled arrays otherwise so callers can detect missing MG metadata.
    """
    with h5py.File(h5_path, 'r') as f:
        gaia_ids  = np.array([g.decode('utf-8').strip() for g in f['metadata']['GaiaDR3_ID'][:]])
        tic_ids   = f['metadata']['tic'][:]
        sectors   = f['metadata']['sector'][:]
        bprp0     = f['metadata']['BPRP0'][:].astype(float)
        bprp0_err = f['metadata']['e_BPRP0'][:].astype(float)
        lengths   = f['length'][:]
        if 'MG_quick' in f['metadata'].keys():
            mg = f['metadata']['MG_quick'][:].astype(float)
        else:
            mg = np.full(len(tic_ids), np.nan)
        if 'MG_quick_err' in f['metadata'].keys():
            mg_err = f['metadata']['MG_quick_err'][:].astype(float)
        else:
            mg_err = np.full(len(tic_ids), np.nan)

    mask = (
        np.array([len(g) > 0 for g in gaia_ids])
        & (tic_ids > 0)
        & (lengths > 0)
        & (~np.isnan(bprp0))
        & (~np.isnan(bprp0_err))
        & (bprp0_err > 0)
    )
    n_total, n_valid = len(mask), int(mask.sum())
    print(f'  {n_valid}/{n_total} entries pass strict filter ({n_total - n_valid} skipped)')
    valid_indices = np.where(mask)[0]
    return (valid_indices, gaia_ids[valid_indices], tic_ids[valid_indices],
            sectors[valid_indices], bprp0[valid_indices], bprp0_err[valid_indices],
            mg[valid_indices], mg_err[valid_indices])


def extract_latents_lazy(ae_model_path, h5_path, valid_indices, tic_ids,
                         hidden_size, direction, mode, pooling_mode,
                         device, batch_size=32):
    """Extract per-sector multiscale latents, lazy H5 reads grouped by star.

    Uses full untruncated sequences with dynamic per-batch padding (pad to the
    longest sequence in each batch, not a global max).
    Metadata is loaded eagerly (small) and passed to every forward call so the
    RNN hidden states match those produced during training.
    """
    from plot_umap_latent import load_model, compute_multiscale_features

    print(f'Loading autoencoder: {ae_model_path}')
    model = load_model(ae_model_path, device, hidden_size=hidden_size,
                       direction=direction, mode=mode, use_flow=True,
                       num_meta_features=13)
    model.eval()

    N = len(valid_indices)
    with h5py.File(h5_path, 'r') as f:
        all_lengths = f['length'][valid_indices].astype(int)
        h5_seq_len  = f['flux'].shape[1]  # padded length in H5

        # Load and standardize metadata eagerly — small array, one row per sector
        meta_grp = f.get('metadata')
        if meta_grp is not None and all(fld in meta_grp for fld in METADATA_FEATURES):
            raw = {fld: meta_grp[fld][valid_indices].astype(np.float32)
                   for fld in METADATA_FEATURES}
            standardizer = MetadataStandardizer(fields=METADATA_FEATURES)
            all_metadata = standardizer.transform(raw)   # (N, num_meta_features)
            print(f'  Loaded metadata: {all_metadata.shape[1]} features')
        else:
            all_metadata = None
            print('  Warning: metadata not found in H5 — latents may differ from training')

    unique_tics = np.unique(tic_ids)
    tic_to_pos  = {t: np.where(tic_ids == t)[0] for t in unique_tics}
    per_sector  = [None] * N

    def _run_buffer(star_buffer, h5_file):
        all_pos    = np.concatenate([pos for pos, _ in star_buffer])
        star_sizes = [len(pos) for pos, _ in star_buffer]
        h5_idxs    = valid_indices[all_pos]
        sort_order = np.argsort(h5_idxs)
        restore    = np.argsort(sort_order)
        sorted_h5  = h5_idxs[sort_order]

        # Dynamic padding: read only up to the longest sequence in this batch
        batch_lengths = all_lengths[all_pos]
        batch_max_len = int(batch_lengths.max())
        read_len      = min(batch_max_len, h5_seq_len)

        flux_s     = h5_file['flux'][sorted_h5, :read_len]
        flux_err_s = h5_file['flux_err'][sorted_h5, :read_len]
        time_s     = h5_file['time'][sorted_h5, :read_len]
        lens_s     = all_lengths[all_pos[sort_order]]

        flux_b     = torch.tensor(flux_s[restore],     dtype=torch.float32, device=device)
        flux_err_b = torch.tensor(flux_err_s[restore], dtype=torch.float32, device=device)
        time_b     = torch.tensor(time_s[restore],     dtype=torch.float32, device=device)
        lens_b     = lens_s[restore]

        if all_metadata is not None:
            meta_b = torch.tensor(all_metadata[all_pos], dtype=torch.float32, device=device)
        else:
            meta_b = None

        B, L = flux_b.shape
        mask_t = torch.zeros(B, L, device=device)
        for j, vlen in enumerate(lens_b):
            mask_t[j, :int(vlen)] = 1.0
        out   = model(torch.stack([flux_b, flux_err_b], dim=-1),
                      time_b.unsqueeze(-1), mask=mask_t,
                      metadata=meta_b, conv_data=None, return_states=True)
        h_fwd = out.get('h_fwd_tensor')
        h_bwd = out.get('h_bwd_tensor')
        hf_sp = torch.split(h_fwd, star_sizes, dim=0) if h_fwd is not None else [None] * len(star_buffer)
        hb_sp = torch.split(h_bwd, star_sizes, dim=0) if h_bwd is not None else [None] * len(star_buffer)
        offset = 0
        for (pos, _), hf, hb in zip(star_buffer, hf_sp, hb_sp):
            for j in range(len(pos)):
                vlen = int(lens_b[offset + j])
                h = torch.cat([hf[j, :vlen, :], hb[j, :vlen, :]], dim=-1) \
                    if (hf is not None and hb is not None) else \
                    (hf[j, :vlen, :] if hf is not None else hb[j, :vlen, :])
                if pooling_mode == 'multiscale':
                    latent = compute_multiscale_features(h, n_segments=4)
                elif pooling_mode == 'mean':
                    latent = h.mean(dim=0)
                else:
                    latent = h[-1, :]
                per_sector[pos[j]] = latent.cpu().numpy()
            offset += len(pos)
        del flux_b, flux_err_b, time_b, meta_b, h_fwd, h_bwd, out, mask_t

    stars_done, star_buffer, buffered_secs = 0, [], 0
    print(f'Extracting latents for {len(unique_tics)} stars ({N} sectors, dynamic padding)...')
    with h5py.File(h5_path, 'r') as h5_file, torch.no_grad():
        for tic in unique_tics:
            pos = tic_to_pos[tic]
            star_buffer.append((pos, tic))
            buffered_secs += len(pos)
            if buffered_secs >= batch_size:
                _run_buffer(star_buffer, h5_file)
                stars_done += len(star_buffer)
                star_buffer, buffered_secs = [], 0
                if stars_done % 500 < batch_size:
                    print(f'  {stars_done}/{len(unique_tics)} stars')
        if star_buffer:
            _run_buffer(star_buffer, h5_file)

    return np.stack(per_sector)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_latent_max(latents, gaia_ids, bprp0, bprp0_err, mg, mg_err):
    """Element-wise max across sectors per Gaia ID."""
    unique_gids = np.unique(gaia_ids)
    star_latents, star_bprp0, star_bprp0_err = [], [], []
    star_mg, star_mg_err = [], []
    for gid in unique_gids:
        m = gaia_ids == gid
        star_latents.append(latents[m].max(axis=0))
        star_bprp0.append(float(bprp0[m][0]))
        star_bprp0_err.append(float(bprp0_err[m][0]))
        star_mg.append(float(mg[m][0]))
        star_mg_err.append(float(mg_err[m][0]))
    return (np.array(star_latents), np.array(star_bprp0),
            np.array(star_bprp0_err), np.array(star_mg),
            np.array(star_mg_err), unique_gids)


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

def build_model(encoder_type, pca_dim, config, device):
    """Reconstruct an AgePredictor or AgePredictorMLP from config."""
    flow_transforms  = config.get('flow_transforms', 8)
    flow_hidden_dims = config.get('flow_hidden_dims', [64, 64])
    use_mg           = config.get('use_mg', False)

    if encoder_type == 'pca':
        return AgePredictor(
            pca_dim=pca_dim,
            flow_transforms=flow_transforms,
            flow_hidden_features=flow_hidden_dims,
            use_mg=use_mg,
        ).to(device)
    else:  # mlp or linear
        input_dim       = config['latent_dim'] if encoder_type == 'pca' else config.get('input_dim', config['latent_dim'])
        mlp_hidden      = [] if encoder_type == 'linear' else config.get('mlp_encoder_hidden', [256, 128])
        aux_loss_weight = config.get('aux_loss_weight', 1.0)
        dropout         = config.get('dropout', 0.1)
        var_reg         = config.get('variance_reg_weight', 0.0)
        return AgePredictorMLP(
            input_dim=input_dim,
            bottleneck_dim=pca_dim,
            mlp_hidden=mlp_hidden,
            flow_transforms=flow_transforms,
            flow_hidden_features=flow_hidden_dims,
            aux_loss_weight=aux_loss_weight,
            use_mg=use_mg,
            dropout=dropout,
            variance_reg_weight=var_reg,
        ).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Age inference for unlabeled stars using a trained k-fold NLE model.')
    parser.add_argument('--nle_dir',     required=True,
                        help='Directory containing kfold_models.pt and config.json')
    parser.add_argument('--model_file',  default=None,
                        help='Model checkpoint filename (default: auto-detect '
                             'full_model.pt, then kfold_models.pt)')
    parser.add_argument('--ae_model',    default=None,
                        help='Autoencoder model.pt (not needed if --load_latents given)')
    parser.add_argument('--h5_path',     default='exop_hosts/lc_data.h5')
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--load_latents', default=None,
                        help='Load pre-extracted per-sector latents from .npz')
    parser.add_argument('--save_latents', default=None,
                        help='Save per-sector latents to .npz for future reuse')
    parser.add_argument('--metadata_csv', default=None,
                        help='CSV with GaiaDR3_ID/BPRP0/BPRP0_err — used to fill NaN '
                             'BPRP0/BPRP0_err in the latents cache (e.g. caches '
                             'written before the host H5 Gaia ID repair).')
    parser.add_argument('--hidden_size',  type=int, default=64)
    parser.add_argument('--direction',    default='bi')
    parser.add_argument('--mode',         default='parallel')
    parser.add_argument('--pooling_mode', default='multiscale')
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--loga_grid_size', type=int, default=LOGA_GRID_DEFAULT_SIZE)
    parser.add_argument('--nle_batch_size', type=int, default=256)
    args = parser.parse_args()

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nle_dir    = Path(args.nle_dir)
    print(f'Device: {device}')

    # ── Load model checkpoint + config ─────────────────────────────────────
    # Auto-detect: prefer full_model.pt (trained on all stars) over kfold_models.pt
    if args.model_file is not None:
        model_path = nle_dir / args.model_file
    elif (nle_dir / 'full_model.pt').exists():
        model_path = nle_dir / 'full_model.pt'
    else:
        model_path = nle_dir / 'kfold_models.pt'

    print(f'\n=== Loading NLE model: {model_path} ===')
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    with open(nle_dir / 'config.json') as fp:
        config = json.load(fp)

    fold_models   = ckpt['fold_models']
    norm_params   = ckpt['normalization_params']
    pca_dim       = ckpt['pca_dim']
    n_folds       = ckpt['n_folds']
    encoder_type  = config.get('encoder_type', 'pca')
    use_mg_only   = bool(config.get('use_mg_only', False))

    # Augment config with norm_params for model construction
    config['input_dim'] = norm_params['input_dim']
    config['latent_dim'] = norm_params['latent_dim']

    print(f'  encoder_type={encoder_type}  pca_dim={pca_dim}  n_folds={n_folds}  '
          f'use_mg_only={use_mg_only}')

    # ── Get per-sector latents ───────────────────────────────────────────────
    if args.load_latents:
        print(f'\n=== Loading latents: {args.load_latents} ===')
        npz       = np.load(args.load_latents, allow_pickle=True)
        # Accept both schemas: predict_ages format ('latents') and
        # plot_umap_latent / kfold_age_inference canonical format ('latent_vectors').
        if 'latent_vectors' in npz.files:
            latents = npz['latent_vectors']
        elif 'latents' in npz.files:
            latents = npz['latents']
        else:
            raise KeyError(f'No latents/latent_vectors in {args.load_latents}; '
                           f'keys = {npz.files}')
        gaia_ids  = npz['gaia_ids'].astype(str)
        bprp0     = npz['bprp0'].astype(float) if 'bprp0' in npz.files \
                    else np.full(len(gaia_ids), np.nan)
        bprp0_err = npz['bprp0_err'].astype(float) if 'bprp0_err' in npz.files \
                    else np.full(len(gaia_ids), np.nan)
        mg        = npz['mg'].astype(float) if 'mg' in npz.files \
                    else np.full(len(gaia_ids), np.nan)
        mg_err    = npz['mg_err'].astype(float) if 'mg_err' in npz.files \
                    else np.full(len(gaia_ids), np.nan)

        # Fill NaN photometry/MG from a metadata CSV when provided. Caches
        # written before the host H5 Gaia ID repair have NaN here because the
        # bad sci-notation IDs failed the original CSV lookup; older caches also
        # predate the MG_quick / MG_quick_err columns.
        n_bad = int(np.isnan(bprp0).sum() + np.isnan(bprp0_err).sum()
                    + np.isnan(mg).sum() + np.isnan(mg_err).sum())
        if n_bad and args.metadata_csv:
            print(f'  filling {n_bad} NaN BPRP0/MG entries from {args.metadata_csv}')
            df = pd.read_csv(args.metadata_csv)
            df['GaiaDR3_ID'] = df['GaiaDR3_ID'].astype(np.int64).astype(str)
            id_to_bprp0     = dict(zip(df['GaiaDR3_ID'], df['BPRP0']))
            id_to_bprp0_err = dict(zip(df['GaiaDR3_ID'], df['BPRP0_err']))
            id_to_mg        = (dict(zip(df['GaiaDR3_ID'], df['MG_quick']))
                               if 'MG_quick' in df.columns else {})
            id_to_mg_err    = (dict(zip(df['GaiaDR3_ID'], df['MG_quick_err']))
                               if 'MG_quick_err' in df.columns else {})
            for i, g in enumerate(gaia_ids):
                if np.isnan(bprp0[i]):
                    bprp0[i]     = id_to_bprp0.get(g, np.nan)
                if np.isnan(bprp0_err[i]):
                    bprp0_err[i] = id_to_bprp0_err.get(g, np.nan)
                if np.isnan(mg[i]):
                    mg[i]        = id_to_mg.get(g, np.nan)
                if np.isnan(mg_err[i]):
                    mg_err[i]    = id_to_mg_err.get(g, np.nan)

        # Drop rows missing the photometry the model was trained on.
        if use_mg_only:
            ok = np.isfinite(mg) & np.isfinite(mg_err) & (mg_err > 0)
            label = 'MG/MG_err'
        else:
            ok = np.isfinite(bprp0) & np.isfinite(bprp0_err) & (bprp0_err > 0)
            label = 'BPRP0/BPRP0_err'
        if (~ok).any():
            print(f'  dropping {(~ok).sum()}/{len(ok)} rows with bad {label}')
            latents, gaia_ids, bprp0, bprp0_err, mg, mg_err = (
                latents[ok], gaia_ids[ok], bprp0[ok], bprp0_err[ok],
                mg[ok], mg_err[ok],
            )
        print(f'  {len(gaia_ids)} sectors, {len(np.unique(gaia_ids))} unique stars, dim={latents.shape[1]}')
    else:
        if args.ae_model is None:
            raise ValueError('--ae_model is required unless --load_latents is given')
        print(f'\n=== Loading H5 metadata: {args.h5_path} ===')
        (valid_indices, gaia_ids, tic_ids, _, bprp0, bprp0_err,
         mg, mg_err) = load_and_filter_h5_metadata(args.h5_path)
        print(f'\n=== Extracting latents ===')
        latents = extract_latents_lazy(
            args.ae_model, args.h5_path, valid_indices, tic_ids,
            args.hidden_size, args.direction, args.mode,
            args.pooling_mode, device, args.batch_size,
        )
        print(f'  {latents.shape[0]} latents, dim={latents.shape[1]}')
        if args.save_latents:
            Path(args.save_latents).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save_latents,
                                latents=latents, gaia_ids=gaia_ids,
                                bprp0=bprp0, bprp0_err=bprp0_err,
                                mg=mg, mg_err=mg_err)
            print(f'Saved latents → {args.save_latents}')

    # ── Aggregate per-sector → one latent per star (latent_max) ─────────────
    print('\n=== Aggregating by Gaia ID (latent_max) ===')
    (star_latents, star_bprp0, star_bprp0_err, star_mg, star_mg_err,
     star_gaia_ids) = aggregate_latent_max(latents, gaia_ids, bprp0, bprp0_err,
                                            mg, mg_err)
    print(f'  {len(star_gaia_ids)} unique stars')

    # ── Optional MG-only swap (must mirror training) ───────────────────────
    if use_mg_only:
        ok = np.isfinite(star_mg) & np.isfinite(star_mg_err) & (star_mg_err > 0)
        if (~ok).any():
            print(f'  --use_mg_only: dropping {(~ok).sum()}/{len(ok)} stars with bad MG/MG_err')
            star_latents   = star_latents[ok]
            star_bprp0     = star_bprp0[ok]
            star_bprp0_err = star_bprp0_err[ok]
            star_mg        = star_mg[ok]
            star_mg_err    = star_mg_err[ok]
            star_gaia_ids  = star_gaia_ids[ok]
        print(f'  --use_mg_only: swapping (BPRP0, BPRP0_err) ← (MG, MG_err) '
              f'[{len(star_gaia_ids)} stars].')
        star_bprp0     = star_mg
        star_bprp0_err = star_mg_err

    # ── Normalise latents ────────────────────────────────────────────────────
    X_mean = norm_params['X_mean']
    X_std  = norm_params['X_std']
    X_norm = (star_latents - X_mean) / (X_std + 1e-8)

    log_bprp0_err = np.log10(np.clip(star_bprp0_err, 1e-10, None)).astype(np.float32)
    log_mg_dummy  = np.zeros(len(star_gaia_ids), dtype=np.float32)

    loga_grid = torch.tensor(
        np.linspace(PRIOR_LOGA_MYR[0], PRIOR_LOGA_MYR[1], args.loga_grid_size),
        dtype=torch.float32, device=device,
    )
    G = args.loga_grid_size
    n = len(star_gaia_ids)

    # ── Posterior inference: ensemble across all K fold models ───────────────
    print(f'\n=== Running age inference ({n_folds}-fold ensemble) ===')
    sum_posterior = np.zeros((n, G), dtype=np.float64)

    for fold_idx, fm in enumerate(fold_models):
        print(f'  Fold {fold_idx + 1}/{n_folds}...')
        model = build_model(encoder_type, pca_dim, config, device)
        model.load_state_dict({k: v.to(device) for k, v in fm['state_dict'].items()})
        model.eval()

        # Apply fold's PCA if encoder_type == 'pca'
        if encoder_type == 'pca' and fm['pca'] is not None:
            X_enc = fm['pca'].transform(X_norm).astype(np.float32)
        else:
            X_enc = X_norm.astype(np.float32)

        fold_posterior = np.zeros((n, G), dtype=np.float64)
        for start in range(0, n, args.nle_batch_size):
            end  = min(start + args.nle_batch_size, n)
            z_t  = torch.tensor(X_enc[start:end],                           device=device)
            b_t  = torch.tensor(star_bprp0[start:end].astype(np.float32),  device=device)
            be_t = torch.tensor(log_bprp0_err[start:end],                  device=device)
            mg_t = torch.tensor(log_mg_dummy[start:end],                   device=device)

            with torch.no_grad():
                # For MLP/linear encoders, run through the learned encoder first
                if encoder_type in ('mlp', 'linear'):
                    z_t = model.encoder(z_t)

                B, D = z_t.shape
                z_exp   = z_t.unsqueeze(1).expand(B, G, D).reshape(B * G, D)
                b_exp   = b_t.unsqueeze(1).expand(B, G).reshape(B * G)
                be_exp  = be_t.unsqueeze(1).expand(B, G).reshape(B * G)
                la_exp  = loga_grid.unsqueeze(0).expand(B, G).reshape(B * G)
                context = torch.stack([la_exp, b_exp, be_exp], dim=1)
                log_lik = model.flow(context).log_prob(z_exp).reshape(B, G)
                # Normalise to posterior
                log_post = log_lik - torch.logsumexp(log_lik, dim=1, keepdim=True)
                fold_posterior[start:end] = log_post.exp().cpu().numpy()

        sum_posterior += fold_posterior
        del model

    # Average posteriors across folds
    avg_posterior = sum_posterior / n_folds

    # ── Extract summary statistics from averaged posterior ───────────────────
    loga_np = loga_grid.cpu().numpy()
    cdf     = avg_posterior.cumsum(axis=1)

    def pct(p):
        idx = np.argmax(cdf >= p, axis=1)
        return loga_np[idx]

    pred_median = pct(0.50)
    pred_p16    = pct(0.16)
    pred_p84    = pct(0.84)
    pred_mean   = (avg_posterior * loga_np[None, :]).sum(axis=1)
    pred_map    = loga_np[avg_posterior.argmax(axis=1)]

    # ── Save ─────────────────────────────────────────────────────────────────
    results = pd.DataFrame({
        'gaia_id':     star_gaia_ids,
        'bprp0':       star_bprp0,
        'pred_median': pred_median,
        'pred_mean':   pred_mean,
        'pred_map':    pred_map,
        'pred_p16':    pred_p16,
        'pred_p84':    pred_p84,
    })
    out_csv = output_dir / 'age_predictions.csv'
    results.to_csv(out_csv, index=False)

    # ── Save run config ───────────────────────────────────────────────────
    run_config = {
        'nle_dir':        str(nle_dir),
        'nle_model_file': str(model_path.name),
        'nle_model_path': str(model_path),
        'ae_model':       args.ae_model,
        'h5_path':        args.h5_path,
        'load_latents':   args.load_latents,
        'save_latents':   args.save_latents,
        'encoder_type':   encoder_type,
        'use_mg_only':    use_mg_only,
        'pca_dim':        pca_dim,
        'n_folds':        n_folds,
        'hidden_size':    args.hidden_size,
        'direction':      args.direction,
        'mode':           args.mode,
        'pooling_mode':   args.pooling_mode,
        'batch_size':     args.batch_size,
        'loga_grid_size': args.loga_grid_size,
        'nle_batch_size': args.nle_batch_size,
        'n_stars':        len(results),
        'nle_training_config': config,
    }
    config_path = output_dir / 'predict_ages_config.json'
    with open(config_path, 'w') as fp:
        json.dump(run_config, fp, indent=2, default=str)

    print(f'\nSaved {len(results)} predictions → {out_csv}')
    print(f'Saved run config → {config_path}')
    print(f'  pred_median: mean={pred_median.mean():.2f}  '
          f'std={pred_median.std():.2f} dex  (log10 age/Myr)')
    print('Done.')


if __name__ == '__main__':
    main()
