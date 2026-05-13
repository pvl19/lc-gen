"""Extract pooled MLP-baseline features for downstream age inference.

For each labeled (gaia_id, sector) in the age CSV, this script:
  1. Forwards the trained MLPGaussianBaseline at multiple k values, taps the
     last hidden layer (128-dim, post-LayerNorm + GELU) for a dense subsample
     of valid j positions.
  2. Averages the per-j features across the k grid (marginalizes out k).
  3. Pools across j with mean + std -> 256-dim per sector.
  4. Writes the canonical save_latents_cache npz format so that
     kfold_age_inference.py --load_latents <out> works without changes.

Used to test whether a same-budget local-window MLP encoder produces age-
informative pooled features comparable to the BiDirectionalMinGRU.
See docs/plans/2026-05-12_mlp-pooled-age-inference.md.

All parameters hardcoded in extract_mlp_latents.sh per project convention.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from lcgen.models.mlp_baseline import MLPGaussianBaseline, build_context_batch
from baseline_comparison import build_index, SequenceStore, METADATA_FEATURES


def find_last_gelu_index(net: torch.nn.Sequential) -> int:
    last = None
    for i, layer in enumerate(net):
        if isinstance(layer, torch.nn.GELU):
            last = i
    if last is None:
        raise RuntimeError('Expected GELU activation in MLP encoder.')
    return last


def load_age_csv(path: str) -> pd.DataFrame:
    """Load the cluster-age CSV; column names match final_pretrain/all_ages.csv."""
    df = pd.read_csv(path)
    df['GaiaDR3_ID'] = df['GaiaDR3_ID'].astype(str)
    df = df.drop_duplicates('GaiaDR3_ID', keep='first').set_index('GaiaDR3_ID')
    return df


def extract_sector_features(mlp, captured, flux, ferr, times, meta,
                            k_grid, n_j_target, C, rng, device):
    """Forward MLP encoder at each k for a shared subsample of valid j; average
    last-hidden features across k. Returns (n_j, D) tensor or None if too short."""
    L = int(flux.shape[0])
    k_max = max(k_grid)
    j_min = k_max + C
    j_max = L - k_max - C
    if j_max <= j_min:
        return None

    n_full = j_max - j_min
    if n_j_target and n_j_target < n_full:
        picks = rng.choice(n_full, size=n_j_target, replace=False)
        picks.sort()
        js = torch.from_numpy(picks.astype(np.int64) + j_min).to(device)
    else:
        js = torch.arange(j_min, j_max, device=device, dtype=torch.long)

    feats_sum = None
    for k in k_grid:
        x, _, _, _ = build_context_batch(
            flux, ferr, times, meta, k=int(k), C=C, js=js,
        )
        _ = mlp.encoder(x)
        h = captured['x']                       # (N, 128) — last hidden
        feats_sum = h.clone() if feats_sum is None else feats_sum + h
    feats = feats_sum / float(len(k_grid))
    return feats


def pool_mean_std(feats: torch.Tensor) -> np.ndarray:
    """Pool (N, D) -> (2D,) by concatenating mean and (unbiased) std along axis 0."""
    mean = feats.mean(dim=0)
    std = feats.std(dim=0, unbiased=False)
    return torch.cat([mean, std], dim=0).cpu().numpy().astype(np.float32)


def save_cache(out_path: Path, latents, ages, bprp0, bprp0_err,
               gaia_ids, tic_ids, sectors, mg, mg_err, mem_prob):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        latent_vectors=latents,
        ages=ages,
        bprp0=bprp0, bprp0_err=bprp0_err,
        mg=mg, mg_err=mg_err, mem_prob=mem_prob,
        gaia_ids=np.asarray(gaia_ids).astype(str),
        tic_ids=np.asarray(tic_ids, dtype=np.int64),
        sectors=np.asarray(sectors, dtype=np.int64),
    )
    print(f'[save] {out_path}')
    print(f'[save] N={len(ages)} sectors, latent_dim={latents.shape[1]}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mlp-path', required=True)
    p.add_argument('--h5-paths', nargs='+', required=True)
    p.add_argument('--age-csv', default=None,
                   help='If provided, restrict extraction to gaia_ids in this CSV and '
                        'fill ages/bprp0/mg columns from it. If omitted, extract over '
                        'all sequences and write NaNs for those columns.')
    p.add_argument('--out-path', required=True)
    p.add_argument('--k-grid', type=int, nargs='+', default=[1, 8, 64, 720])
    p.add_argument('--n-j-per-sector', type=int, default=1024)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log-every', type=int, default=200)
    args = p.parse_args()

    device = torch.device(args.device)

    # ---- Load MLP checkpoint and set up hook on last GELU ----
    ckpt = torch.load(args.mlp_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    C = int(cfg['C'])
    mlp = MLPGaussianBaseline(
        C=C,
        num_meta_features=int(cfg['num_meta_features']),
        hidden_dims=tuple(cfg['hidden_dims']),
        context_dim=int(cfg['context_dim']),
    ).to(device)
    mlp.load_state_dict(ckpt['model_state'])
    mlp.eval()
    print(f'[load] MLP from {args.mlp_path}')
    print(f'[load] C={C}  hidden_dims={cfg["hidden_dims"]}  '
          f'context_dim={cfg["context_dim"]}  num_meta={cfg["num_meta_features"]}')

    last_gelu_idx = find_last_gelu_index(mlp.encoder.net)
    captured = {}

    def hook(_module, _inputs, output):
        captured['x'] = output

    handle = mlp.encoder.net[last_gelu_idx].register_forward_hook(hook)
    print(f'[hook] tapping encoder.net[{last_gelu_idx}] '
          f'(out_features={mlp.encoder.net[last_gelu_idx - 1].weight.shape[0]})')

    # ---- Index sequences (optionally filter by age CSV) ----
    full_index = build_index([Path(p) for p in args.h5_paths])
    if args.age_csv:
        df = load_age_csv(args.age_csv)
        print(f'[load] {len(df)} labeled stars in {args.age_csv}')
        labeled_ids = set(df.index.tolist())
        labeled_idx = [i for i, e in enumerate(full_index)
                       if str(e['gaia_id']) in labeled_ids]
        print(f'[index] {len(labeled_idx)} labeled sequences across H5 files '
              f'(out of {len(full_index)})')
    else:
        df = None
        labeled_idx = list(range(len(full_index)))
        print(f'[index] {len(labeled_idx)} sequences (no age-CSV filter)')

    # ---- Iterate and extract ----
    store = SequenceStore(full_index, use_metadata=True, device=device)
    rng = np.random.default_rng(args.seed)

    latents, ages, bprp0, bprp0_err = [], [], [], []
    mg, mg_err, mem_prob = [], [], []
    gaia_out, tic_out, sec_out = [], [], []

    t0 = time.time()
    n_done = n_skipped = 0
    with torch.no_grad():
        for entry, flux, ferr, times, meta in store.iter_chunks(
                labeled_idx, shuffle_chunks=False, shuffle_within=False):
            feats = extract_sector_features(
                mlp, captured, flux, ferr, times, meta,
                k_grid=args.k_grid, n_j_target=args.n_j_per_sector,
                C=C, rng=rng, device=device,
            )
            if feats is None:
                n_skipped += 1
                continue
            pooled = pool_mean_std(feats)
            gid = str(entry['gaia_id'])
            latents.append(pooled)
            if df is not None and gid in df.index:
                row = df.loc[gid]
                ages.append(float(row.get('age_Myr', np.nan)))
                bprp0.append(float(row.get('BPRP0', np.nan)))
                bprp0_err.append(float(row.get('BPRP0_err', np.nan)))
                mg.append(float(row.get('MG_quick', np.nan)))
                mg_err.append(float(row.get('MG_quick_err', np.nan)))
                mem_prob.append(float(row.get('mem_prob_val', np.nan)))
            else:
                ages.append(np.nan)
                bprp0.append(np.nan); bprp0_err.append(np.nan)
                mg.append(np.nan); mg_err.append(np.nan)
                mem_prob.append(np.nan)
            gaia_out.append(gid)
            tic_out.append(int(entry.get('tic', 0)))
            sec_out.append(int(entry.get('sector', 0)))
            n_done += 1
            if args.log_every and n_done % args.log_every == 0:
                dt = time.time() - t0
                rate = n_done / dt if dt else 0.0
                print(f'  [extract] {n_done}/{len(labeled_idx)} '
                      f'({rate:.1f} seq/s, {n_skipped} skipped too-short)')

    store.close()
    handle.remove()
    print(f'[done] {n_done} sectors extracted, {n_skipped} skipped (too short)')

    save_cache(
        Path(args.out_path),
        latents=np.stack(latents).astype(np.float32),
        ages=np.asarray(ages, dtype=np.float64),
        bprp0=np.asarray(bprp0, dtype=np.float64),
        bprp0_err=np.asarray(bprp0_err, dtype=np.float64),
        gaia_ids=gaia_out, tic_ids=tic_out, sectors=sec_out,
        mg=np.asarray(mg, dtype=np.float64),
        mg_err=np.asarray(mg_err, dtype=np.float64),
        mem_prob=np.asarray(mem_prob, dtype=np.float64),
    )


if __name__ == '__main__':
    main()
