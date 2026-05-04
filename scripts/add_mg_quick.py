"""Add `MG_quick` and `MG_quick_err` to H5 metadata groups and CSV files.

Definitions:
    MG_quick     = G0 + 5 * log10(parallax_mas) - 10
    MG_quick_err = sqrt( G0_err^2 + (5 / (parallax_mas * ln10))^2 * parallax_error^2 )

equivalently MG = G0 + 5*log10(parallax_mas/100). Negative or zero parallaxes
(noisy Gaia detections) yield NaN for both quantities.

Targets are passed positionally; the script auto-detects type by suffix:
    .h5  → opens metadata group, creates / replaces `MG_quick` and `MG_quick_err`
    .csv → reads/writes the file in place, adding/replacing columns

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

    # Apply (refuses if target already exists; pass --force to overwrite):
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


def compute_mg_err(parallax_mas: np.ndarray, parallax_err_mas: np.ndarray,
                   g0_err: np.ndarray) -> np.ndarray:
    """1-sigma propagation: σ_MG² = σ_G0² + (5/(p·ln10))² · σ_p².

    NaN where parallax ≤ 0 or any input is non-finite. Treats negative G0_err /
    parallax_error as NaN (some catalogs encode missing as -9999 or similar).
    """
    p  = np.asarray(parallax_mas,     dtype=np.float64)
    pe = np.asarray(parallax_err_mas, dtype=np.float64)
    ge = np.asarray(g0_err,           dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = ((p > 0) & np.isfinite(p)
          & np.isfinite(pe) & (pe >= 0)
          & np.isfinite(ge) & (ge >= 0))
    coef = 5.0 / (p[ok] * np.log(10.0))
    out[ok] = np.sqrt(ge[ok] ** 2 + (coef * pe[ok]) ** 2)
    return out


def _summarize(arr: np.ndarray, label: str):
    n = len(arr)
    n_nan = int(np.isnan(arr).sum())
    finite = arr[~np.isnan(arr)]
    rng = f'[{finite.min():.3f}, {finite.max():.3f}]' if finite.size else 'empty'
    print(f'  {label}: n={n}  finite={n - n_nan}  NaN={n_nan}  range={rng}')


def update_h5(path: Path, dry_run: bool, force: bool):
    print(f'\n--- {path} (h5) ---')
    if not path.exists():
        print('  missing, skipping.')
        return

    with h5py.File(path, 'r') as f:
        md = f['metadata']
        needed = ('G0', 'parallax', 'G0_err', 'parallax_error')
        missing = [k for k in needed if k not in md.keys()]
        if missing:
            print(f'  missing {missing} in metadata; skipping.')
            return
        existing_mg     = 'MG_quick'     in md.keys()
        existing_mg_err = 'MG_quick_err' in md.keys()
        g0       = md['G0'][:]
        parallax = md['parallax'][:]
        g0_err   = md['G0_err'][:]
        plx_err  = md['parallax_error'][:]

    mg     = compute_mg(g0, parallax)
    mg_err = compute_mg_err(parallax, plx_err, g0_err)
    _summarize(mg,     'computed MG_quick    ')
    _summarize(mg_err, 'computed MG_quick_err')

    if (existing_mg or existing_mg_err) and not force:
        present = [k for k, exists in
                   [('MG_quick', existing_mg), ('MG_quick_err', existing_mg_err)] if exists]
        print(f'  {present} already exist — pass --force to overwrite. Skipping.')
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
        for name, data in (('MG_quick', mg), ('MG_quick_err', mg_err)):
            if name in md.keys():
                del md[name]
            md.create_dataset(name, data=data.astype(np.float64))
    print(f'  wrote MG_quick + MG_quick_err into {path}')


def update_csv(path: Path, dry_run: bool, force: bool):
    print(f'\n--- {path} (csv) ---')
    if not path.exists():
        print('  missing, skipping.')
        return

    df = pd.read_csv(path)
    needed = ('G0', 'parallax', 'G0_err', 'parallax_error')
    missing = [k for k in needed if k not in df.columns]
    if missing:
        print(f'  missing {missing} in columns; skipping. cols={list(df.columns)}')
        return
    existing_mg     = 'MG_quick'     in df.columns
    existing_mg_err = 'MG_quick_err' in df.columns

    mg     = compute_mg(df['G0'].to_numpy(), df['parallax'].to_numpy())
    mg_err = compute_mg_err(df['parallax'].to_numpy(),
                            df['parallax_error'].to_numpy(),
                            df['G0_err'].to_numpy())
    _summarize(mg,     'computed MG_quick    ')
    _summarize(mg_err, 'computed MG_quick_err')

    if (existing_mg or existing_mg_err) and not force:
        present = [k for k, exists in
                   [('MG_quick', existing_mg), ('MG_quick_err', existing_mg_err)] if exists]
        print(f'  {present} already exist — pass --force to overwrite. Skipping.')
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

    df['MG_quick']     = mg
    df['MG_quick_err'] = mg_err
    df.to_csv(path, index=False)
    print(f'  wrote MG_quick + MG_quick_err into {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='H5 and/or CSV files to update')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='Overwrite existing MG_quick / MG_quick_err column/dataset')
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
