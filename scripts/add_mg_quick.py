"""Add `MG_quick` (absolute G magnitude) to H5 metadata groups and CSV files.

Definition:
    MG_quick = G0 + 5 * log10(parallax_mas) - 10

equivalently MG = G0 + 5*log10(parallax_mas/100). Negative or zero parallaxes
(noisy Gaia detections) yield NaN.

Targets are passed positionally; the script auto-detects type by suffix:
    .h5  → opens metadata group, creates / replaces the `MG_quick` dataset
    .csv → reads/writes the file in place, adding/replacing an `MG_quick` column

Both forms back up the file once before the first rewrite (`.bak` for CSV,
`.mg_quick_backup.h5` sidecar copy for H5).

Usage:
    # Dry-run (no writes):
    python scripts/add_mg_quick.py \\
        final_pretrain/timeseries_pretrain.h5 \\
        final_pretrain/timeseries_exop_hosts.h5 \\
        final_pretrain/all_ages.csv \\
        final_pretrain/host_all_metadata.csv \\
        --dry-run

    # Apply (refuses if MG_quick already exists; pass --force to overwrite):
    python scripts/add_mg_quick.py <files...>
    python scripts/add_mg_quick.py <files...> --force
"""
import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def compute_mg(g0: np.ndarray, parallax_mas: np.ndarray) -> np.ndarray:
    """MG = G0 + 5*log10(parallax_mas) - 10. NaN where parallax_mas <= 0."""
    g0 = np.asarray(g0, dtype=np.float64)
    p = np.asarray(parallax_mas, dtype=np.float64)
    out = np.full_like(g0, np.nan)
    ok = (p > 0) & np.isfinite(p) & np.isfinite(g0)
    out[ok] = g0[ok] + 5.0 * np.log10(p[ok]) - 10.0
    return out


def _summarize(mg: np.ndarray, label: str):
    n = len(mg)
    n_nan = int(np.isnan(mg).sum())
    finite = mg[~np.isnan(mg)]
    rng = f'[{finite.min():.2f}, {finite.max():.2f}]' if finite.size else 'empty'
    print(f'  {label}: n={n}  finite={n - n_nan}  NaN={n_nan}  range={rng}')


def update_h5(path: Path, dry_run: bool, force: bool):
    print(f'\n--- {path} (h5) ---')
    if not path.exists():
        print('  missing, skipping.')
        return

    with h5py.File(path, 'r') as f:
        md = f['metadata']
        if 'G0' not in md.keys() or 'parallax' not in md.keys():
            print('  missing G0 or parallax in metadata; skipping.')
            return
        existing = 'MG_quick' in md.keys()
        g0 = md['G0'][:]
        parallax = md['parallax'][:]

    mg = compute_mg(g0, parallax)
    _summarize(mg, 'computed MG_quick')

    if existing and not force:
        print('  MG_quick already exists — pass --force to overwrite. Skipping.')
        return
    if dry_run:
        print('  [dry-run] no changes written.')
        return

    backup = path.with_suffix('.mg_quick_backup' + path.suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f'  backup → {backup}')
    else:
        print(f'  backup already exists → {backup} (not overwriting)')

    with h5py.File(path, 'r+') as f:
        md = f['metadata']
        if 'MG_quick' in md.keys():
            del md['MG_quick']
        md.create_dataset('MG_quick', data=mg.astype(np.float64))
    print(f'  wrote MG_quick into {path}')


def update_csv(path: Path, dry_run: bool, force: bool):
    print(f'\n--- {path} (csv) ---')
    if not path.exists():
        print('  missing, skipping.')
        return

    df = pd.read_csv(path)
    if 'G0' not in df.columns or 'parallax' not in df.columns:
        print(f'  missing G0 or parallax in columns; skipping. cols={list(df.columns)}')
        return
    existing = 'MG_quick' in df.columns

    mg = compute_mg(df['G0'].to_numpy(), df['parallax'].to_numpy())
    _summarize(mg, 'computed MG_quick')

    if existing and not force:
        print('  MG_quick already exists — pass --force to overwrite. Skipping.')
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

    df['MG_quick'] = mg
    df.to_csv(path, index=False)
    print(f'  wrote MG_quick into {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='H5 and/or CSV files to update')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='Overwrite existing MG_quick column/dataset')
    args = ap.parse_args()

    for f in args.files:
        p = Path(f)
        if p.suffix == '.h5':
            update_h5(p, args.dry_run, args.force)
        elif p.suffix == '.csv':
            update_csv(p, args.dry_run, args.force)
        else:
            print(f'\n--- {p} --- unknown extension; skipping.')


if __name__ == '__main__':
    main()
