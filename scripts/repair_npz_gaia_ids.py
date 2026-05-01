"""Repair scientific-notation GaiaDR3_IDs in cached npz files via TIC ID lookup.

When `final_pretrain/timeseries_exop_hosts.h5` was first written, most
GaiaDR3_IDs were serialized as scientific-notation strings (precision-lossy via
float64). Downstream caches built from that H5 — host latent caches, UMAP
embedding npz files, etc. — inherited the bad IDs.

The H5 itself has since been fixed by `scripts/fix_host_h5_gaia_ids.py`. This
script repairs any stale npz that has both `tic_ids` and `gaia_ids` by joining
on TIC ID against `final_pretrain/host_all_metadata.csv`, where Gaia IDs are
stored as proper int64.

Usage:
    # Dry-run a single file:
    python scripts/repair_npz_gaia_ids.py path/to/file.npz --dry-run

    # Repair in place (saves a .bak copy):
    python scripts/repair_npz_gaia_ids.py path/to/file.npz

    # Multiple files:
    python scripts/repair_npz_gaia_ids.py a.npz b.npz c.npz
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METADATA_CSV = Path('final_pretrain/host_all_metadata.csv')


def load_tic_to_gaia(csv_path: Path) -> dict[int, str]:
    df = pd.read_csv(csv_path)
    # Both columns are int64 in this CSV; cast to be explicit.
    tic = df['TIC_ID'].astype('int64').values
    gid = df['GaiaDR3_ID'].astype('int64').values
    return {int(t): str(int(g)) for t, g in zip(tic, gid)}


def repair_file(path: Path, tic_to_gaia: dict[int, str], dry_run: bool):
    print(f'\n--- {path} ---')
    if not path.exists():
        print('  missing, skipping.')
        return

    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    if 'gaia_ids' not in keys or 'tic_ids' not in keys:
        print(f'  no gaia_ids/tic_ids fields; skipping (keys={keys})')
        return

    old_gids = d['gaia_ids'].astype(str)
    tic_ids = d['tic_ids'].astype('int64')
    n = len(old_gids)

    new_gids = np.empty(n, dtype=old_gids.dtype)
    n_sci = 0
    n_changed = 0
    n_missing_tic = 0
    sample_changes = []
    for i, (g, t) in enumerate(zip(old_gids, tic_ids)):
        is_sci = 'e' in g.lower()
        if is_sci:
            n_sci += 1
        proper = tic_to_gaia.get(int(t))
        if proper is None:
            n_missing_tic += 1
            new_gids[i] = g  # leave alone
            continue
        if g != proper:
            n_changed += 1
            if len(sample_changes) < 3:
                sample_changes.append((i, int(t), g, proper))
        new_gids[i] = proper

    print(f'  rows: {n}')
    print(f'  with sci notation:        {n_sci}')
    print(f'  TIC not found in CSV:     {n_missing_tic}')
    print(f'  rows whose ID will change:{n_changed}')
    for i, t, old, new in sample_changes:
        print(f'    row {i}: tic={t}  {old!r:>40}  →  {new!r}')

    if n_changed == 0:
        print('  nothing to do.')
        return

    if dry_run:
        print('  [dry-run] no changes written.')
        return

    backup = path.with_suffix(path.suffix + '.bak')
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f'  backup → {backup}')
    else:
        print(f'  backup already exists → {backup} (not overwriting)')

    # Rewrite the file with the same key set, only gaia_ids replaced.
    save_dict = {k: d[k] for k in keys}
    save_dict['gaia_ids'] = new_gids
    np.savez(path, **save_dict)
    print(f'  rewrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='npz files to repair')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--metadata_csv', type=Path, default=METADATA_CSV)
    args = ap.parse_args()

    if not args.metadata_csv.exists():
        sys.exit(f'Missing metadata CSV: {args.metadata_csv}')
    tic_to_gaia = load_tic_to_gaia(args.metadata_csv)
    print(f'Loaded {len(tic_to_gaia)} TIC→Gaia mappings from {args.metadata_csv}')

    for f in args.files:
        repair_file(Path(f), tic_to_gaia, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
