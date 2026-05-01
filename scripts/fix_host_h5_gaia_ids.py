"""Repair `final_pretrain/timeseries_exop_hosts.h5` GaiaDR3_ID column.

The H5 was written with most GaiaDR3_IDs serialized as scientific-notation
strings (e.g. '5.157183324996790e+18'), which lost precision when cast through
float64 (Gaia IDs need 19 digits, float64 has ~16). This blocks crossmatch
against archive_ages_default.csv on Gaia ID.

This script restores the correct integer-string Gaia IDs by joining on TIC ID
against final_pretrain/host_all_metadata.csv, where TIC and Gaia are stored
as proper int64.

Writes a sidecar .id_repair_backup.npz with {row, old_id, new_id} so the
change is fully auditable and reversible.

Usage:
    python scripts/fix_host_h5_gaia_ids.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


H5_PATH       = Path('final_pretrain/timeseries_exop_hosts.h5')
METADATA_CSV  = Path('final_pretrain/host_all_metadata.csv')
BACKUP_PATH   = H5_PATH.with_suffix('.gaia_id_repair_backup.npz')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Print stats and proposed changes without writing.')
    args = parser.parse_args()

    if not H5_PATH.exists():
        sys.exit(f'Missing: {H5_PATH}')
    if not METADATA_CSV.exists():
        sys.exit(f'Missing: {METADATA_CSV}')

    df = pd.read_csv(METADATA_CSV)
    if df['TIC_ID'].duplicated().any():
        sys.exit('host_all_metadata.csv has duplicate TIC_IDs — refusing to proceed.')
    tic_to_gaia = dict(zip(df['TIC_ID'].astype(np.int64),
                           df['GaiaDR3_ID'].astype(np.int64)))
    print(f'Loaded {len(tic_to_gaia)} TIC→Gaia mappings from {METADATA_CSV}')

    with h5py.File(H5_PATH, 'r') as f:
        h5_tics  = f['metadata']['tic'][:]
        h5_gids  = f['metadata']['GaiaDR3_ID'][:]
        ds_dtype = f['metadata']['GaiaDR3_ID'].dtype
    n = len(h5_tics)
    print(f'Host H5 has {n} rows, GaiaDR3_ID dtype = {ds_dtype}')

    missing = [t for t in np.unique(h5_tics) if int(t) not in tic_to_gaia]
    if missing:
        sys.exit(f'{len(missing)} unique TICs in H5 not found in metadata CSV; '
                 f'first few: {missing[:5]}. Refusing to proceed.')

    # Build new Gaia ID column
    new_gids_int = np.array([tic_to_gaia[int(t)] for t in h5_tics], dtype=np.int64)
    new_gids_str = np.array([str(g) for g in new_gids_int], dtype=ds_dtype)

    # Diff stats
    old_gids_str = np.array([g.decode().strip() if isinstance(g, bytes) else str(g).strip()
                             for g in h5_gids])
    changed = []
    sci_n = plain_n = 0
    for i, (o, n_int) in enumerate(zip(old_gids_str, new_gids_int)):
        is_sci = ('e' in o.lower())
        if is_sci:
            sci_n += 1
            try:
                if int(float(o)) != n_int:
                    changed.append(i)
            except Exception:
                changed.append(i)
        else:
            plain_n += 1
            if int(o) != n_int:
                changed.append(i)
    print(f'\n=== Diff summary ===')
    print(f'  rows with scientific-notation IDs: {sci_n}')
    print(f'  rows with plain-int IDs:           {plain_n}')
    print(f'  rows whose ID will change:         {len(changed)}')
    if changed:
        idx = changed[0]
        print(f'  e.g. row {idx}: tic={h5_tics[idx]}  '
              f'old="{old_gids_str[idx]}"  new="{new_gids_int[idx]}"')

    if args.dry_run:
        print('\n[dry-run] No changes written.')
        return

    # Sidecar backup (small): row idx + old + new IDs
    np.savez(
        BACKUP_PATH,
        row=np.arange(n, dtype=np.int64),
        tic=h5_tics,
        old_gaia_id_raw=h5_gids,           # bytes, exact original
        new_gaia_id_int=new_gids_int,
    )
    print(f'\nSidecar backup → {BACKUP_PATH} ({BACKUP_PATH.stat().st_size / 1e6:.2f} MB)')

    print(f'Updating GaiaDR3_ID dataset in {H5_PATH} ...')
    with h5py.File(H5_PATH, 'r+') as f:
        f['metadata']['GaiaDR3_ID'][...] = new_gids_str
    print('Done.')

    # Verify
    with h5py.File(H5_PATH, 'r') as f:
        verify = f['metadata']['GaiaDR3_ID'][:5]
    print(f'Sample updated IDs: {[v.decode() for v in verify]}')


if __name__ == '__main__':
    main()
