"""Produce a sector_stats CSV that mirrors the latent split for sectors 97/98.

Reads the existing per-(gaia, sector) stats CSV and, for rows where
sector ∈ {97, 98}, looks up the underlying light curve in the H5 files,
recomputes flux_skew / flux_kurt on each half (matching the latent split:
trim_edges only at the original sector boundaries, midpoint split of the
remaining valid range), and writes one row per half.

All other columns (ages, bprp0, umap_x/y, num_flares, total_flare_ed,
lit_Prot, tars_Prot, ...) are COPIED from the original sector-97/98 row to
both halves — the user explicitly does not want per-sector astrophysical stats
recomputed. A new `subsector` column distinguishes the halves:

    -1 = unchanged row (not split)
     0 = first  half of an original sector-97/98 light curve
     1 = second half of an original sector-97/98 light curve

This is the same `subsector` convention used by `latents_*_s97s98_split.npz`
and the merged latents bank, so the new stats CSV joins to the new UMAP
output on (gaia_id, sector, subsector).
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats


TARGET_SECTORS = (97, 98)


def build_h5_index(h5_paths):
    """Map (gaia_id, sector) -> (h5_path, row_idx). Only target-sector rows are
    indexed (everything else is irrelevant). Earlier H5 paths win on conflict.
    """
    index = {}
    for path in h5_paths:
        with h5py.File(path, 'r') as f:
            sectors = f['metadata']['sector'][:]
            target_mask = np.isin(sectors, np.asarray(TARGET_SECTORS, dtype=sectors.dtype))
            target_idx = np.where(target_mask)[0]
            if len(target_idx) == 0:
                continue
            gaia_b = f['metadata']['GaiaDR3_ID'][target_idx]
        for h5_i, gaia in zip(target_idx, gaia_b):
            gid = gaia.decode('utf-8').strip() if isinstance(gaia, (bytes, bytearray)) else str(gaia).strip()
            sec = int(sectors[h5_i])
            key = (gid, sec)
            if key in index:
                continue
            index[key] = (path, int(h5_i))
    return index


def compute_half_moments(h5_path, h5_idx, trim_edges, min_post):
    """Return (skew0, kurt0, skew1, kurt1) for the two halves, or None if the
    row is too short (<2*min_post post-trim samples). Mirrors the latent split:
    take [trim:trim+valid], split at midpoint.
    """
    with h5py.File(h5_path, 'r') as f:
        L = int(f['length'][h5_idx])
        valid = L - 2 * trim_edges
        if valid < 2 * min_post:
            return None
        mid = valid // 2
        a_start = trim_edges
        a_stop  = trim_edges + mid
        b_start = trim_edges + mid
        b_stop  = trim_edges + valid
        flux_a = f['flux'][h5_idx, a_start:a_stop]
        flux_b = f['flux'][h5_idx, b_start:b_stop]
    flux_a = flux_a[np.isfinite(flux_a)]
    flux_b = flux_b[np.isfinite(flux_b)]
    if flux_a.size < 4 or flux_b.size < 4:
        return None
    return (
        float(stats.skew(flux_a, bias=False)),
        float(stats.kurtosis(flux_a, bias=False, fisher=True)),
        float(stats.skew(flux_b, bias=False)),
        float(stats.kurtosis(flux_b, bias=False, fisher=True)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in_csv', default='data/sector_stats.csv')
    p.add_argument('--out_csv', default='data/sector_stats_s97s98_split.csv')
    p.add_argument('--h5_paths', nargs='+',
                   default=['final_pretrain/timeseries_pretrain.h5',
                            'final_pretrain/timeseries_exop_hosts.h5',
                            'final_pretrain/timeseries_thickdisk.h5'])
    p.add_argument('--trim_edges', type=int, default=10,
                   help='Must match the latent extraction (default 10).')
    p.add_argument('--min_post', type=int, default=32,
                   help='Minimum samples per half (default 32, matches latent extraction).')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f'Refusing to overwrite {out_path} without --overwrite')

    print(f'Reading {in_path} ...')
    df = pd.read_csv(in_path)
    print(f'  {len(df)} rows, columns: {list(df.columns)}')

    if 'sector' not in df.columns or 'gaia_id' not in df.columns:
        raise SystemExit("input CSV must have at least 'gaia_id' and 'sector' columns")

    # Add subsector column to the unchanged portion (default -1).
    df = df.copy()
    df['subsector'] = np.int64(-1)

    target_mask = df['sector'].isin(TARGET_SECTORS)
    df_target = df.loc[target_mask].copy()
    df_kept   = df.loc[~target_mask].copy()
    print(f'  {len(df_target)} rows in sectors {list(TARGET_SECTORS)} '
          f'(removing & replacing with split halves)')

    print('\nBuilding H5 index for target sectors ...')
    h5_paths = [Path(p) for p in args.h5_paths]
    index = build_h5_index(h5_paths)
    print(f'  indexed {len(index)} unique (gaia_id, sector) pairs in target sectors')

    print(f'\nRecomputing flux_skew / flux_kurt on halves (trim_edges={args.trim_edges}, '
          f'min_post={args.min_post}) ...')
    split_rows = []
    n_no_h5 = 0
    n_too_short = 0
    n_split = 0
    # Use itertuples (NOT iterrows): iterrows returns a single-dtype Series per
    # row, which upcasts int64 gaia_ids to float64 (Gaia IDs are 19-digit;
    # float64 has only ~15-17 sig figs, so the join silently corrupts).
    gids_int = df_target['gaia_id'].astype(np.int64).values
    secs_int = df_target['sector'].astype(np.int64).values
    for pos, row_t in enumerate(df_target.itertuples(index=False)):
        gid = str(int(gids_int[pos]))
        sec = int(secs_int[pos])
        key = (gid, sec)
        if key not in index:
            n_no_h5 += 1
            continue
        h5_path, h5_idx = index[key]
        result = compute_half_moments(h5_path, h5_idx,
                                      trim_edges=args.trim_edges,
                                      min_post=args.min_post)
        if result is None:
            n_too_short += 1
            continue
        skew0, kurt0, skew1, kurt1 = result
        # _asdict preserves per-column dtypes (unlike iterrows). Keep gaia_id /
        # tic_id / sector as int64 explicitly.
        row_dict = row_t._asdict()
        row_dict['gaia_id'] = int(gids_int[pos])
        row_dict['sector']  = sec
        if 'tic_id' in row_dict and pd.notna(row_dict['tic_id']):
            row_dict['tic_id'] = int(row_dict['tic_id'])
        for half_idx, (sk, ku) in enumerate([(skew0, kurt0), (skew1, kurt1)]):
            new_row = dict(row_dict)
            new_row['flux_skew'] = sk
            new_row['flux_kurt'] = ku
            new_row['subsector'] = np.int64(half_idx)
            split_rows.append(new_row)
        n_split += 1

    print(f'  split {n_split} rows -> {len(split_rows)} half-rows')
    if n_no_h5:
        print(f'  WARNING: {n_no_h5} sector-{TARGET_SECTORS} rows had no matching H5 entry — dropped')
    if n_too_short:
        print(f'  WARNING: {n_too_short} sector-{TARGET_SECTORS} rows too short to split — dropped')

    if not split_rows:
        raise SystemExit('No split rows produced — aborting.')

    df_split = pd.DataFrame(split_rows, columns=df.columns)
    out = pd.concat([df_kept, df_split], ignore_index=True)

    # Preserve column order: original columns + subsector at the end.
    cols = [c for c in df.columns if c != 'subsector'] + ['subsector']
    out = out[cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f'\nWrote {len(out)} rows -> {out_path}')
    print(f'  unchanged rows: {len(df_kept)}  '
          f'(subsector=-1: {(out["subsector"] == -1).sum()})')
    print(f'  split half-rows: {len(df_split)}  '
          f'(subsector=0: {(out["subsector"] == 0).sum()}, '
          f'subsector=1: {(out["subsector"] == 1).sum()})')


if __name__ == '__main__':
    main()
