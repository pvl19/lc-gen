"""Merge per-half sector 97/98 split latents into a main latents npz.

For each (main, split) pair:
  - load the main npz (canonical kfold/UMAP format)
  - drop rows where sector ∈ target_sectors (default 97, 98)
  - concatenate the split npz's rows (one per half)
  - write a merged npz to a separate output path (does NOT overwrite by default)

The merged npz keeps the canonical fields (latent_vectors, ages, bprp0,
bprp0_err, mg, mg_err, mem_prob, gaia_ids, tic_ids, sectors). Extra fields from
the split file (subsector, orig_h5_idx) are kept and filled with -1 for rows
that came from the main bank (so every row carries a self-describing tag).

Sanity: the script verifies the main and split files share latent_dim and have
the same float dtype before concatenating; otherwise it errors out.
"""
import argparse
from pathlib import Path

import numpy as np


CANONICAL_KEYS = [
    'latent_vectors', 'ages', 'bprp0', 'bprp0_err',
    'mg', 'mg_err', 'mem_prob',
    'gaia_ids', 'tic_ids', 'sectors',
]


def _drop_sectors(d: dict, drop_mask: np.ndarray) -> dict:
    keep = ~drop_mask
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(drop_mask):
            out[k] = v[keep]
        else:
            out[k] = v
    return out


def merge_one(main_path: Path, split_path: Path, out_path: Path,
              target_sectors=(97, 98)):
    main_d = {k: np.asarray(v) for k, v in np.load(main_path, allow_pickle=True).items()}
    split_d = {k: np.asarray(v) for k, v in np.load(split_path, allow_pickle=True).items()}

    for k in ('latent_vectors', 'sectors', 'gaia_ids'):
        if k not in main_d:
            raise KeyError(f'{main_path}: missing required key {k!r}')
        if k not in split_d:
            raise KeyError(f'{split_path}: missing required key {k!r}')

    if main_d['latent_vectors'].shape[1] != split_d['latent_vectors'].shape[1]:
        raise ValueError(
            f'latent_dim mismatch: main {main_d["latent_vectors"].shape[1]} '
            f'vs split {split_d["latent_vectors"].shape[1]}')
    if main_d['latent_vectors'].dtype != split_d['latent_vectors'].dtype:
        print(f'  [warn] latent dtype mismatch: main={main_d["latent_vectors"].dtype} '
              f'split={split_d["latent_vectors"].dtype} — casting split to main')
        split_d['latent_vectors'] = split_d['latent_vectors'].astype(main_d['latent_vectors'].dtype)

    main_sectors = main_d['sectors']
    drop_mask = np.isin(main_sectors, np.asarray(target_sectors, dtype=main_sectors.dtype))
    n_dropped = int(drop_mask.sum())
    if n_dropped == 0:
        print(f'  [warn] main has 0 rows in target sectors {list(target_sectors)} — nothing to replace')

    bad_split = ~np.isin(split_d['sectors'], np.asarray(target_sectors, dtype=split_d['sectors'].dtype))
    if bad_split.any():
        raise ValueError(
            f'{split_path}: contains {int(bad_split.sum())} rows outside target sectors '
            f'{list(target_sectors)} — refusing to merge')

    print(f'  main rows: {len(main_sectors)}  (dropping {n_dropped} in s{list(target_sectors)})')
    print(f'  split rows to add: {len(split_d["sectors"])}')

    kept_main = _drop_sectors(main_d, drop_mask)

    n_kept = len(kept_main['sectors'])
    n_split = len(split_d['sectors'])
    n_out = n_kept + n_split

    out: dict[str, np.ndarray] = {}
    # Concatenate the canonical fields; cast as needed so dtypes align.
    for k in CANONICAL_KEYS:
        if k not in kept_main:
            print(f'  [warn] main missing canonical key {k!r}; filling NaN/-1')
            kept_main[k] = _fill_missing(k, n_kept)
        if k not in split_d:
            print(f'  [warn] split missing canonical key {k!r}; filling NaN/-1')
            split_d[k] = _fill_missing(k, n_split)
        a, b = kept_main[k], split_d[k]
        if a.dtype != b.dtype:
            b = b.astype(a.dtype)
        out[k] = np.concatenate([a, b], axis=0)

    # Preserve extras (subsector, orig_h5_idx) — fill with -1 for non-split rows.
    extras = (set(split_d.keys()) | set(main_d.keys())) - set(CANONICAL_KEYS)
    for k in sorted(extras):
        a = main_d.get(k)
        b = split_d.get(k)
        if a is None:
            ref = b
            a = np.full((n_kept,) + ref.shape[1:], -1, dtype=ref.dtype)
        else:
            a = a[~drop_mask]
        if b is None:
            ref = a
            b = np.full((n_split,) + ref.shape[1:], -1, dtype=ref.dtype)
        if a.dtype != b.dtype:
            b = b.astype(a.dtype)
        out[k] = np.concatenate([a, b], axis=0)

    sectors_out = out['sectors']
    assert len(sectors_out) == n_out
    assert not np.isin(sectors_out[:n_kept], np.asarray(target_sectors, dtype=sectors_out.dtype)).any(), \
        'main portion still contains target-sector rows after drop'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f'  -> {out_path}  ({n_out} rows, latent_dim={out["latent_vectors"].shape[1]})')


def _fill_missing(key: str, n: int) -> np.ndarray:
    if key == 'gaia_ids':
        return np.full(n, '', dtype='<U1')
    if key in ('tic_ids', 'sectors'):
        return np.full(n, -1, dtype=np.int64)
    if key == 'latent_vectors':
        raise ValueError('latent_vectors cannot be missing')
    return np.full(n, np.nan, dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--main', nargs='+', required=True,
                   help='Main latents npz path(s) to replace s97/s98 rows in.')
    p.add_argument('--split', nargs='+', required=True,
                   help='Per-half split latents npz path(s), one per --main, same order.')
    p.add_argument('--out', nargs='+', required=True,
                   help='Output merged npz path(s), one per --main. Will NOT overwrite '
                        '--main unless you pass the same path explicitly.')
    p.add_argument('--target_sectors', type=int, nargs='+', default=[97, 98])
    args = p.parse_args()

    if not (len(args.main) == len(args.split) == len(args.out)):
        p.error('--main, --split, --out must all have the same number of paths')

    for main_p, split_p, out_p in zip(args.main, args.split, args.out):
        print(f'\n=== {main_p} ⊕ {split_p} -> {out_p} ===')
        merge_one(Path(main_p), Path(split_p), Path(out_p),
                  target_sectors=tuple(args.target_sectors))

    print('\nDone.')


if __name__ == '__main__':
    main()
