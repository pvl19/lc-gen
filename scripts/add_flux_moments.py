"""Add `flux_skew` and `flux_kurt` to H5 metadata groups, and write a per-row
CSV (one row per star × sector) with the same values.

Definitions (per light curve / per H5 row, computed over the valid length):
    flux_skew = scipy.stats.skew    (bias=False, finite values only)
    flux_kurt = scipy.stats.kurtosis(bias=False, fisher=True; i.e. excess kurtosis)

Rows with fewer than 4 finite flux samples produce NaN.

CSV columns: GaiaDR3_ID, TIC_ID, sector, flux_skew, flux_kurt.
The per-row CSV is appended to across multiple H5 inputs.

Usage:
    # Dry-run (no writes):
    python scripts/add_flux_moments.py \\
        final_pretrain/timeseries_pretrain.h5 \\
        final_pretrain/timeseries_exop_hosts.h5 \\
        --csv final_pretrain/flux_moments.csv \\
        --dry-run

    # Apply (refuses if H5 datasets exist; pass --force to overwrite):
    python scripts/add_flux_moments.py <h5_files...> --csv <out.csv>
    python scripts/add_flux_moments.py <h5_files...> --csv <out.csv> --force

H5 files are backed up once via a `.flux_moments_backup.h5` sidecar copy. The
output CSV is overwritten on each run (use --force to allow overwrite).
"""
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats


def compute_moments_h5(path: Path) -> dict:
    """Return dict with arrays: gaia_id, tic, sector, flux_skew, flux_kurt.

    Iterates row-by-row to avoid loading the full (N, T_max) flux array.
    """
    with h5py.File(path, 'r') as f:
        flux_ds = f['flux']
        length  = f['length'][:]
        ids_b   = f['metadata/GaiaDR3_ID'][:]
        tic     = f['metadata/tic'][:]
        sector  = f['metadata/sector'][:]
        n = length.shape[0]
        skew = np.full(n, np.nan, dtype=np.float64)
        kurt = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            L = int(length[i])
            if L <= 0:
                continue
            x = flux_ds[i, :L]
            x = x[np.isfinite(x)]
            if x.size < 4:
                continue
            skew[i] = stats.skew(x, bias=False)
            kurt[i] = stats.kurtosis(x, bias=False, fisher=True)
            if (i + 1) % 5000 == 0:
                print(f'    {i + 1}/{n} rows processed')
    ids = np.array([s.decode('utf-8').strip() if isinstance(s, (bytes, bytearray)) else str(s).strip()
                    for s in ids_b])
    return {'GaiaDR3_ID': ids, 'TIC_ID': tic, 'sector': sector,
            'flux_skew': skew, 'flux_kurt': kurt}


def _summarize(arr: np.ndarray, label: str):
    n = len(arr)
    n_nan = int(np.isnan(arr).sum())
    finite = arr[~np.isnan(arr)]
    rng = f'[{finite.min():.3f}, {finite.max():.3f}]' if finite.size else 'empty'
    print(f'  {label}: n={n}  finite={n - n_nan}  NaN={n_nan}  range={rng}')


def update_h5(path: Path, dry_run: bool, force: bool) -> dict | None:
    """Write flux_skew/flux_kurt into the H5; return per-row arrays for CSV use."""
    print(f'\n--- {path} (h5) ---')
    if not path.exists():
        print('  missing, skipping.')
        return None

    with h5py.File(path, 'r') as f:
        md = f['metadata']
        existing_skew = 'flux_skew' in md.keys()
        existing_kurt = 'flux_kurt' in md.keys()

    print('  computing per-row moments...')
    rec = compute_moments_h5(path)
    _summarize(rec['flux_skew'], 'computed flux_skew')
    _summarize(rec['flux_kurt'], 'computed flux_kurt')

    if (existing_skew or existing_kurt) and not force:
        present = [k for k, e in [('flux_skew', existing_skew),
                                  ('flux_kurt', existing_kurt)] if e]
        print(f'  {present} already exist — pass --force to overwrite H5 datasets. '
              f'Returning computed values for CSV anyway.')
        return rec

    if dry_run:
        print('  [dry-run] no H5 changes written.')
        return rec

    backup = path.with_suffix('.flux_moments_backup' + path.suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f'  backup → {backup}')
    else:
        print(f'  backup already exists → {backup} (not overwriting)')

    with h5py.File(path, 'r+') as f:
        md = f['metadata']
        for name in ('flux_skew', 'flux_kurt'):
            if name in md.keys():
                del md[name]
            md.create_dataset(name, data=rec[name].astype(np.float64))
    print(f'  wrote flux_skew + flux_kurt into {path}')
    return rec


def write_csv(out_path: Path, recs: list[dict], dry_run: bool, force: bool):
    print(f'\n--- {out_path} (csv) ---')
    if not recs:
        print('  no H5 records produced; skipping CSV.')
        return
    if out_path.exists() and not force and not dry_run:
        print(f'  {out_path} exists — pass --force to overwrite. Skipping.')
        return

    df = pd.concat([pd.DataFrame(r) for r in recs], ignore_index=True)
    df = df[['GaiaDR3_ID', 'TIC_ID', 'sector', 'flux_skew', 'flux_kurt']]
    print(f'  total rows: {len(df)}')

    if dry_run:
        print('  [dry-run] no CSV written.')
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'  wrote {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('h5_files', nargs='+', help='H5 files to update / read flux from')
    ap.add_argument('--csv', required=True, help='Output CSV path (one row per star × sector)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='Overwrite existing flux_skew / flux_kurt H5 datasets and CSV')
    args = ap.parse_args()

    recs = []
    for f in args.h5_files:
        p = Path(f)
        if p.suffix != '.h5':
            print(f'\n--- {p} --- not an .h5 file; skipping.')
            continue
        rec = update_h5(p, args.dry_run, args.force)
        if rec is not None:
            recs.append(rec)

    write_csv(Path(args.csv), recs, args.dry_run, args.force)


if __name__ == '__main__':
    main()
