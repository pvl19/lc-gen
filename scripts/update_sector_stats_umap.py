"""Update umap_x / umap_y in a split-sector_stats CSV from a UMAP embedding npz.

Reads:
  - the split sector_stats CSV (must have a `subsector` column)
  - the UMAP output npz (embedding, gaia_ids, sectors — order matches its input)
  - the merged latents npz files used as the UMAP input (each contributes a
    `subsector` array, concatenated in input order to align with UMAP rows)

Joins on (gaia_id, sector, subsector) and writes the embedding into umap_x /
umap_y. Rows in the CSV with no matching UMAP row are left unchanged.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in_csv', default='data/sector_stats_s97s98_split.csv')
    p.add_argument('--out_csv', default=None,
                   help='Default: overwrite --in_csv in place.')
    p.add_argument('--umap_npz', required=True,
                   help='UMAP output npz (embedding, gaia_ids, sectors).')
    p.add_argument('--latent_npz', nargs='+', required=True,
                   help='Merged latents npz files in the SAME ORDER they were '
                        'concatenated into the UMAP input (each must have a '
                        '`subsector` field). The script verifies the total row '
                        'count matches the UMAP embedding length.')
    args = p.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv) if args.out_csv else in_path
    print(f'Reading {in_path} ...')
    df = pd.read_csv(in_path)
    if 'subsector' not in df.columns:
        raise SystemExit("input CSV is missing 'subsector' column — "
                         "did you forget to run split_sector_stats_s97_98.py?")
    print(f'  {len(df)} rows')

    print(f'\nReading UMAP npz: {args.umap_npz}')
    u = np.load(args.umap_npz, allow_pickle=True)
    emb = np.asarray(u['embedding'])
    u_gaia = np.asarray(u['gaia_ids']).astype(str)
    u_sec = np.asarray(u['sectors']).astype(np.int64)
    print(f'  embedding shape={emb.shape}')

    print('\nReading merged latents npz files (for subsector alignment):')
    sub_parts = []
    gaia_parts = []
    sec_parts = []
    for path in args.latent_npz:
        d = np.load(path, allow_pickle=True)
        if 'subsector' not in d:
            raise SystemExit(f'{path} has no `subsector` field')
        sub_parts.append(np.asarray(d['subsector']).astype(np.int64))
        gaia_parts.append(np.asarray(d['gaia_ids']).astype(str))
        sec_parts.append(np.asarray(d['sectors']).astype(np.int64))
        print(f'  {path}: n={len(sub_parts[-1])}')

    l_sub = np.concatenate(sub_parts)
    l_gaia = np.concatenate(gaia_parts)
    l_sec = np.concatenate(sec_parts)
    if len(l_sub) != len(emb):
        raise SystemExit(
            f'merged-latents total rows ({len(l_sub)}) != UMAP embedding rows '
            f'({len(emb)}) — pass the same files in the same order plot_umap.sh used')

    # Sanity: the UMAP npz's own gaia_ids / sectors must agree with the
    # concatenated latent files, position-by-position. If they don't, the
    # latent files we were handed are out of order vs. the UMAP run.
    if not np.array_equal(u_gaia, l_gaia):
        n_mis = int((u_gaia != l_gaia).sum())
        raise SystemExit(
            f'gaia_id alignment mismatch between UMAP npz and latent npzs '
            f'({n_mis}/{len(emb)} rows differ) — check --latent_npz order')
    if not np.array_equal(u_sec, l_sec):
        n_mis = int((u_sec != l_sec).sum())
        raise SystemExit(
            f'sector alignment mismatch between UMAP npz and latent npzs '
            f'({n_mis}/{len(emb)} rows differ)')

    # Build the (gaia_id, sector, subsector) -> (umap_x, umap_y) map.
    keys = list(zip(l_gaia, l_sec.tolist(), l_sub.tolist()))
    if len(set(keys)) != len(keys):
        n_dup = len(keys) - len(set(keys))
        print(f'  WARNING: {n_dup} duplicate (gaia, sector, subsector) keys in '
              f'latents — last occurrence wins')
    lookup = {k: (float(emb[i, 0]), float(emb[i, 1])) for i, k in enumerate(keys)}

    # Look up each CSV row.
    print(f'\nJoining CSV on (gaia_id, sector, subsector) ...')
    gids = df['gaia_id'].astype(np.int64).astype(str).values
    secs = df['sector'].astype(np.int64).values
    subs = df['subsector'].astype(np.int64).values

    new_x = df['umap_x'].astype(float).values.copy()
    new_y = df['umap_y'].astype(float).values.copy()
    n_hit = 0
    n_miss = 0
    miss_examples = []
    for i in range(len(df)):
        key = (gids[i], int(secs[i]), int(subs[i]))
        v = lookup.get(key)
        if v is None:
            n_miss += 1
            if len(miss_examples) < 5:
                miss_examples.append(key)
            continue
        new_x[i] = v[0]
        new_y[i] = v[1]
        n_hit += 1
    df['umap_x'] = new_x
    df['umap_y'] = new_y

    print(f'  updated  : {n_hit}/{len(df)} rows')
    print(f'  no match : {n_miss}/{len(df)} rows (umap_x/y left unchanged)')
    if miss_examples:
        print(f'  first miss keys (gaia_id, sector, subsector):')
        for k in miss_examples:
            print(f'    {k}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
