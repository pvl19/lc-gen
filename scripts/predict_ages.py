"""Run age inference on unlabeled stars using a trained full NLE model.

Extracts latents from an H5 file using the autoencoder. Stars are grouped so
all sectors for a star are processed together (no duplicated forward passes).
H5 data is read lazily per batch — peak RAM is ~batch_size sectors at a time,
not the full dataset.

Strict filtering: entries missing GaiaDR3_ID, BPRP0, e_BPRP0, tic, or with
zero sequence length are skipped — no NaN substitution or default values.

Usage (first time — extract latents from autoencoder):
    python scripts/predict_ages.py \\
        --nle_model  output/full_nle/full_nle_model.pt \\
        --ae_model   output/performance_tests/cf_final/model.pt \\
        --h5_path    exop_hosts/lc_data.h5 \\
        --output_dir output/exop_host_ages \\
        --save_latents output/exop_host_ages/latents.npz

Usage (subsequent runs — skip autoencoder):
    python scripts/predict_ages.py \\
        --nle_model    output/full_nle/full_nle_model.pt \\
        --load_latents output/exop_host_ages/latents.npz \\
        --output_dir   output/exop_host_ages
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from kfold_age_inference import AgePredictor, PRIOR_LOGA_MYR, LOGA_GRID_DEFAULT_SIZE


# ---------------------------------------------------------------------------
# Strict metadata loading + filtering
# ---------------------------------------------------------------------------

def load_and_filter_h5_metadata(h5_path: str):
    """Read metadata from H5 and apply strict filtering.

    Skips any entry with:
      - empty GaiaDR3_ID string
      - tic == 0
      - length == 0  (no actual data)
      - NaN BPRP0 or e_BPRP0
      - e_BPRP0 <= 0

    Returns valid_indices (H5 row indices) and the filtered metadata arrays.
    No NaN substitution or default values are applied.
    """
    with h5py.File(h5_path, 'r') as f:
        gaia_ids  = np.array([g.decode('utf-8').strip() for g in f['metadata']['GaiaDR3_ID'][:]])
        tic_ids   = f['metadata']['tic'][:]
        sectors   = f['metadata']['sector'][:]
        bprp0     = f['metadata']['BPRP0'][:].astype(float)
        bprp0_err = f['metadata']['e_BPRP0'][:].astype(float)
        lengths   = f['length'][:]

    mask = (
        np.array([len(g) > 0 for g in gaia_ids])  # non-empty Gaia ID
        & (tic_ids > 0)                             # valid TIC
        & (lengths > 0)                             # has actual data
        & (~np.isnan(bprp0))                        # valid colour
        & (~np.isnan(bprp0_err))                    # valid colour error
        & (bprp0_err > 0)                           # positive colour error
    )

    n_total = len(mask)
    n_valid = int(mask.sum())
    print(f'  {n_valid} / {n_total} entries pass strict filter ({n_total - n_valid} skipped)')

    valid_indices = np.where(mask)[0]
    return (
        valid_indices,
        gaia_ids[valid_indices],
        tic_ids[valid_indices],
        sectors[valid_indices],
        bprp0[valid_indices],
        bprp0_err[valid_indices],
    )


# ---------------------------------------------------------------------------
# Memory-safe latent extraction (star-grouped, lazy H5 reads)
# ---------------------------------------------------------------------------

def extract_latents_by_star_lazy(ae_model_path, h5_path, valid_indices,
                                  tic_ids, hidden_size, direction, mode,
                                  max_length, pooling_mode, device,
                                  batch_size=32):
    """Extract per-sector latents grouped by star, with lazy H5 reads.

    Mirrors kfold_age_inference.extract_latents_by_star but avoids the upfront
    load of the full dataset. Only `length` is read at startup (~80 KB); flux /
    flux_err / time are fetched from the open H5 file per batch (~batch_size
    sectors at a time, e.g. 32 × 8192 × 3 × 4 B ≈ 3 MB peak).

    Stars are kept whole across batches so all sectors for a given star are
    processed in a single forward pass — no duplicated effort.

    Returns:
        per_sector_latents: (N, D) array, same order as valid_indices
    """
    from plot_umap_latent import load_model, compute_multiscale_features

    print(f'Loading autoencoder: {ae_model_path}')
    model = load_model(
        ae_model_path, device,
        hidden_size=hidden_size, direction=direction, mode=mode,
        use_flow=True, num_meta_features=0,
    )
    model.eval()

    N = len(valid_indices)

    # Read only lengths upfront — tiny (~80 KB for 20k entries)
    with h5py.File(h5_path, 'r') as f:
        all_lengths = f['length'][valid_indices]  # shape (N,), dtype int32

    actual_max = min(max_length, 16384)

    # Map each local position (0..N-1) to its TIC; group positions by TIC
    unique_tics = np.unique(tic_ids)
    tic_to_pos  = {t: np.where(tic_ids == t)[0] for t in unique_tics}

    per_sector_latents = [None] * N

    def _run_buffer(star_buffer, h5_file):
        """Forward pass for a list of (local_positions, tic) pairs.

        Reads H5 lazily for only the rows in this batch.
        """
        all_local_pos = np.concatenate([pos for pos, _ in star_buffer])
        star_sizes    = [len(pos) for pos, _ in star_buffer]

        # Map local positions → original H5 row indices; sort for efficient H5 read
        h5_idxs    = valid_indices[all_local_pos]
        sort_order = np.argsort(h5_idxs)
        sorted_h5  = h5_idxs[sort_order]
        restore    = np.argsort(sort_order)

        flux_s     = h5_file['flux'][sorted_h5, :actual_max]
        flux_err_s = h5_file['flux_err'][sorted_h5, :actual_max]
        time_s     = h5_file['time'][sorted_h5, :actual_max]
        lens_s     = np.minimum(all_lengths[all_local_pos[sort_order]], actual_max)

        # Restore original (star-grouped) order
        flux_b     = torch.tensor(flux_s[restore],     dtype=torch.float32, device=device)
        flux_err_b = torch.tensor(flux_err_s[restore], dtype=torch.float32, device=device)
        time_b     = torch.tensor(time_s[restore],     dtype=torch.float32, device=device)
        lens_b     = lens_s[restore]

        B, L = flux_b.shape
        mask_t = torch.zeros(B, L, device=device)
        for j, vlen in enumerate(lens_b):
            mask_t[j, :int(vlen)] = 1.0

        x_in = torch.stack([flux_b, flux_err_b], dim=-1)
        t_in = time_b.unsqueeze(-1)

        out   = model(x_in, t_in, mask=mask_t, metadata=None,
                      conv_data=None, return_states=True)
        h_fwd = out.get('h_fwd_tensor')  # (B, L, H) or None
        h_bwd = out.get('h_bwd_tensor')  # (B, L, H) or None

        h_fwd_split = torch.split(h_fwd, star_sizes, dim=0) if h_fwd is not None else [None] * len(star_buffer)
        h_bwd_split = torch.split(h_bwd, star_sizes, dim=0) if h_bwd is not None else [None] * len(star_buffer)

        offset = 0
        for (pos, _), hf, hb in zip(star_buffer, h_fwd_split, h_bwd_split):
            for j in range(len(pos)):
                vlen = int(lens_b[offset + j])
                if hf is not None and hb is not None:
                    h = torch.cat([hf[j, :vlen, :], hb[j, :vlen, :]], dim=-1)
                elif hf is not None:
                    h = hf[j, :vlen, :]
                else:
                    h = hb[j, :vlen, :]

                if pooling_mode == 'multiscale':
                    latent = compute_multiscale_features(h, n_segments=4)
                elif pooling_mode == 'mean':
                    latent = h.mean(dim=0)
                elif pooling_mode == 'final':
                    latent = h[-1, :]
                else:
                    raise ValueError(f'Unknown pooling_mode: {pooling_mode}')

                per_sector_latents[pos[j]] = latent.cpu().numpy()
            offset += len(pos)

        del flux_b, flux_err_b, time_b, h_fwd, h_bwd, out, mask_t

    print(f'Extracting latents for {len(unique_tics)} stars '
          f'({N} sectors total, pooling={pooling_mode}, batch_size~{batch_size})...')

    stars_done    = 0
    star_buffer   = []
    buffered_secs = 0

    with h5py.File(h5_path, 'r') as h5_file:
        with torch.no_grad():
            for tic in unique_tics:
                pos = tic_to_pos[tic]
                star_buffer.append((pos, tic))
                buffered_secs += len(pos)

                if buffered_secs >= batch_size:
                    _run_buffer(star_buffer, h5_file)
                    stars_done   += len(star_buffer)
                    star_buffer   = []
                    buffered_secs = 0
                    if stars_done % 500 < batch_size:
                        print(f'  {stars_done}/{len(unique_tics)} stars processed')

            if star_buffer:
                _run_buffer(star_buffer, h5_file)

    return np.stack(per_sector_latents)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_latent_max_by_gaia(latents, gaia_ids, bprp0, bprp0_err):
    """Element-wise max across sectors per Gaia ID."""
    unique_gids    = np.unique(gaia_ids)
    star_latents   = []
    star_bprp0     = []
    star_bprp0_err = []
    for gid in unique_gids:
        mask = gaia_ids == gid
        star_latents.append(latents[mask].max(axis=0))
        star_bprp0.append(float(bprp0[mask][0]))
        star_bprp0_err.append(float(bprp0_err[mask][0]))
    return (np.array(star_latents),
            np.array(star_bprp0),
            np.array(star_bprp0_err),
            unique_gids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nle_model',    required=True,
                        help='Path to full_nle_model.pt')
    parser.add_argument('--ae_model',     default=None,
                        help='Autoencoder model.pt (not needed if --load_latents given)')
    parser.add_argument('--h5_path',      default='exop_hosts/lc_data.h5',
                        help='H5 file with light curves (not needed if --load_latents given)')
    parser.add_argument('--output_dir',   default='output/exop_host_ages')
    parser.add_argument('--load_latents', default=None,
                        help='Load pre-extracted per-sector latents from .npz')
    parser.add_argument('--save_latents', default=None,
                        help='Save per-sector latents to .npz for future reuse')
    # Autoencoder architecture — must match the model used to build the latents cache
    parser.add_argument('--hidden_size',  type=int, default=64)
    parser.add_argument('--direction',    default='bi')
    parser.add_argument('--mode',         default='parallel')
    parser.add_argument('--max_length',   type=int, default=8192)
    parser.add_argument('--pooling_mode', default='multiscale')
    parser.add_argument('--batch_size',   type=int, default=32,
                        help='Target number of sectors per AE forward pass')
    parser.add_argument('--loga_grid_size', type=int, default=LOGA_GRID_DEFAULT_SIZE)
    parser.add_argument('--nle_batch_size', type=int, default=256,
                        help='Batch size for NLE posterior evaluation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load NLE model ──────────────────────────────────────────────────────
    print(f'\n=== Loading NLE model: {args.nle_model} ===')
    ckpt = torch.load(args.nle_model, map_location='cpu', weights_only=False)
    pca              = ckpt['pca']
    norm_params      = ckpt['normalization_params']
    pca_dim          = ckpt['pca_dim']
    use_mg           = ckpt.get('use_mg', False)
    flow_transforms  = ckpt.get('flow_transforms', 12)
    flow_hidden_dims = ckpt.get('flow_hidden_dims', [64, 164])

    nle = AgePredictor(
        pca_dim=pca_dim,
        flow_transforms=flow_transforms,
        flow_hidden_features=flow_hidden_dims,
        use_mg=use_mg,
    ).to(device)
    nle.load_state_dict(ckpt['state_dict'])
    nle.eval()
    print(f'  pca_dim={pca_dim}  flow_transforms={flow_transforms}  use_mg={use_mg}')

    # ── Get per-sector latents ──────────────────────────────────────────────
    if args.load_latents:
        print(f'\n=== Loading latents: {args.load_latents} ===')
        npz       = np.load(args.load_latents, allow_pickle=True)
        latents   = npz['latents']
        gaia_ids  = npz['gaia_ids'].astype(str)
        bprp0     = npz['bprp0'].astype(float)
        bprp0_err = npz['bprp0_err'].astype(float)
        print(f'  {len(gaia_ids)} entries  ({len(np.unique(gaia_ids))} unique Gaia IDs)')
    else:
        if args.ae_model is None:
            raise ValueError('--ae_model is required unless --load_latents is given')

        print(f'\n=== Loading H5 metadata: {args.h5_path} ===')
        valid_indices, gaia_ids, tic_ids, sectors, bprp0, bprp0_err = \
            load_and_filter_h5_metadata(args.h5_path)

        print(f'\n=== Extracting latents (lazy H5, batch_size={args.batch_size}) ===')
        latents = extract_latents_by_star_lazy(
            args.ae_model, args.h5_path, valid_indices, tic_ids,
            args.hidden_size, args.direction, args.mode,
            args.max_length, args.pooling_mode, device,
            batch_size=args.batch_size,
        )
        print(f'  Extracted {latents.shape[0]} latents, dim={latents.shape[1]}')

        if args.save_latents:
            Path(args.save_latents).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.save_latents,
                latents=latents, gaia_ids=gaia_ids,
                bprp0=bprp0,     bprp0_err=bprp0_err,
            )
            print(f'Saved latents → {args.save_latents}')

    # ── Aggregate per-sector → one latent per star (latent_max) ────────────
    print('\n=== Aggregating by Gaia ID (latent_max) ===')
    star_latents, star_bprp0, star_bprp0_err, star_gaia_ids = aggregate_latent_max_by_gaia(
        latents, gaia_ids, bprp0, bprp0_err,
    )
    print(f'  {len(star_gaia_ids)} unique stars')

    # ── Normalise → PCA ─────────────────────────────────────────────────────
    X_norm = (star_latents - norm_params['mean']) / (norm_params['std'] + 1e-8)
    X_pca  = pca.transform(X_norm).astype(np.float32)

    log_bprp0_err = np.log10(np.clip(star_bprp0_err, 1e-10, None)).astype(np.float32)
    log_mg_dummy  = np.zeros(len(star_gaia_ids), dtype=np.float32)

    loga_grid = torch.tensor(
        np.linspace(PRIOR_LOGA_MYR[0], PRIOR_LOGA_MYR[1], args.loga_grid_size),
        dtype=torch.float32, device=device,
    )

    # ── Posterior inference in batches ──────────────────────────────────────
    print('\n=== Running age inference ===')
    stat_keys = ('median', 'mean', 'map', 'p16', 'p84')
    all_stats = {k: [] for k in stat_keys}

    n = len(star_gaia_ids)
    for start in range(0, n, args.nle_batch_size):
        end  = min(start + args.nle_batch_size, n)
        z_t  = torch.tensor(X_pca[start:end],                             device=device)
        b_t  = torch.tensor(star_bprp0[start:end].astype(np.float32),    device=device)
        be_t = torch.tensor(log_bprp0_err[start:end],                    device=device)
        mg_t = torch.tensor(log_mg_dummy[start:end],                     device=device)

        with torch.no_grad():
            stats = nle.predict_stats(z_t, b_t, be_t, mg_t, loga_grid)

        for k in stat_keys:
            all_stats[k].append(stats[k].cpu().numpy())

        if (start // args.nle_batch_size) % 20 == 0:
            print(f'  {end}/{n}')

    for k in stat_keys:
        all_stats[k] = np.concatenate(all_stats[k])

    # ── Save ─────────────────────────────────────────────────────────────────
    results = pd.DataFrame({
        'gaia_id':     star_gaia_ids,
        'bprp0':       star_bprp0,
        'pred_median': all_stats['median'],   # log10(age/Myr)
        'pred_mean':   all_stats['mean'],
        'pred_map':    all_stats['map'],
        'pred_p16':    all_stats['p16'],
        'pred_p84':    all_stats['p84'],
    })
    out_csv = output_dir / 'age_predictions.csv'
    results.to_csv(out_csv, index=False)

    print(f'\nSaved {len(results)} predictions → {out_csv}')
    print(f'  pred_median: mean={all_stats["median"].mean():.2f}  '
          f'std={all_stats["median"].std():.2f} dex  (log10 age/Myr)')
    print('Done.')


if __name__ == '__main__':
    main()
