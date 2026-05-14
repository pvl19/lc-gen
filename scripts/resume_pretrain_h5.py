"""Resume an interrupted build_pretrain_h5.py run.

Re-runs Pass 1 (scan_pickles) to deterministically rebuild the same index,
verifies that already-written rows in the target h5 match that index, then
resumes Pass 2 writing from the first row whose `length` is still 0.

Usage:
    python scripts/resume_pretrain_h5.py \
        --pickle_dir final_pretrain_robust \
        --meta_dir   final_pretrain \
        --out_dir    final_pretrain_robust \
        --target     exop_hosts
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from build_pretrain_h5 import (
    STR_FIELDS, NUM_FIELDS_CSV, NUM_FIELDS_PICKLE,
    scan_pickles,
)


def resume_write(h5_path, index, meta_lookup, host_meta_lookup):
    with h5py.File(h5_path, 'r') as f:
        existing_lengths = f['length'][:]
        existing_tics = f['metadata/tic'][:]
        existing_sectors = f['metadata/sector'][:]

    total = len(index)
    if total != len(existing_lengths):
        raise RuntimeError(
            f'index size {total} != h5 rows {len(existing_lengths)}. '
            'Pickle/metadata changed since original run — cannot safely resume.'
        )

    first_missing = int(np.argmax(existing_lengths == 0)) if (existing_lengths == 0).any() else total
    if first_missing == total:
        print('  Nothing to resume — all rows already written.')
        return

    print(f'  First missing row: {first_missing} / {total}')

    # Verify the last filled row matches the deterministic index ordering
    if first_missing > 0:
        check_i = first_missing - 1
        path, key, _ = index[check_i]
        with open(path, 'rb') as f:
            data = pickle.load(f)
        lc = data[key]
        idx_tic = int(lc['TIC_ID'])
        idx_sector = int(lc['sector'])
        del data
        if idx_tic != existing_tics[check_i] or idx_sector != existing_sectors[check_i]:
            raise RuntimeError(
                f'Verification failed at row {check_i}: '
                f'h5 has tic={existing_tics[check_i]}, sector={existing_sectors[check_i]} '
                f'but index has tic={idx_tic}, sector={idx_sector}. '
                'Index ordering does not match — cannot safely resume.'
            )
        print(f'  Verified row {check_i}: tic={idx_tic}, sector={idx_sector}')

    # Build by_file restricted to row_i >= first_missing,
    # preserving the same pickle-file iteration order as build_pretrain_h5.write_entries.
    by_file = defaultdict(list)
    for row_i, (path, key, _) in enumerate(index):
        if row_i < first_missing:
            continue
        by_file[path].append((row_i, key))

    written = 0
    to_write = total - first_missing
    with h5py.File(h5_path, 'r+') as h5_file:
        for path, rows in by_file.items():
            print(f'  Loading {path} ({len(rows)} rows to write)...')
            with open(path, 'rb') as f:
                data = pickle.load(f)

            for row_i, key in rows:
                lc = data[key]
                tic = int(lc['TIC_ID'])
                is_exop = lc['metadata']['exop_host'] == 1

                meta_row = meta_lookup[tic] if tic in meta_lookup else host_meta_lookup[tic]

                flux     = np.asarray(lc['flux'],     dtype=np.float32)
                flux_err = np.asarray(lc['flux_err'], dtype=np.float32)
                time     = np.asarray(lc['time'],     dtype=np.float64)

                time = time - time[0]

                bad = ~np.isfinite(flux_err)
                if bad.any():
                    median_err = np.nanmedian(flux_err)
                    flux_err[bad] = median_err if np.isfinite(median_err) else 0.0

                L = len(flux)
                h5_file['flux'][row_i, :L]     = flux
                h5_file['flux_err'][row_i, :L] = flux_err
                h5_file['time'][row_i, :L]     = time
                h5_file['length'][row_i]        = L
                h5_file['metadata/tic'][row_i]       = tic
                h5_file['metadata/sector'][row_i]    = int(lc['sector'])
                h5_file['metadata/exop_host'][row_i] = int(is_exop)

                for field in STR_FIELDS:
                    val = meta_row.get(field, '')
                    h5_file[f'metadata/{field}'][row_i] = str(val).encode('utf-8') if pd.notna(val) else b''
                for field in NUM_FIELDS_CSV:
                    val = meta_row.get(field, np.nan)
                    h5_file[f'metadata/{field}'][row_i] = float(val) if pd.notna(val) else np.nan
                for field in NUM_FIELDS_PICKLE:
                    val = lc['metadata'].get(field, np.nan)
                    try:
                        h5_file[f'metadata/{field}'][row_i] = float(val)
                    except (TypeError, ValueError):
                        h5_file[f'metadata/{field}'][row_i] = np.nan

                written += 1
                if written % 1000 == 0:
                    print(f'    {written}/{to_write}')

            del data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pickle_dir', type=Path, required=True)
    ap.add_argument('--meta_dir',   type=Path, required=True)
    ap.add_argument('--out_dir',    type=Path, required=True)
    ap.add_argument('--target', choices=['pretrain', 'exop_hosts', 'both'], default='both')
    args = ap.parse_args()

    pickle_files = [args.pickle_dir / f'lc_data_p{i}.pickle' for i in range(1, 8)]
    meta_csv      = args.meta_dir / 'metadata.csv'
    host_meta_csv = args.meta_dir / 'host_all_metadata.csv'

    print('Loading metadata...')
    meta_df = pd.read_csv(meta_csv)
    meta_df['TIC_ID'] = meta_df['TIC_ID'].astype(int)
    meta_lookup = {row['TIC_ID']: row for _, row in meta_df.iterrows()}
    print(f'  {len(meta_lookup)} unique TICs in metadata.csv')

    host_meta_df = pd.read_csv(host_meta_csv)
    host_meta_df['TIC_ID'] = host_meta_df['TIC_ID'].astype(int)
    host_meta_lookup = {row['TIC_ID']: row for _, row in host_meta_df.iterrows()}
    print(f'  {len(host_meta_lookup)} unique TICs in host_all_metadata.csv')

    print('\nPass 1: scanning pickle files to rebuild index...')
    pretrain_index, exop_index = scan_pickles(pickle_files, meta_lookup, host_meta_lookup)
    print(f'  pretrain index: {len(pretrain_index)}')
    print(f'  exop index:     {len(exop_index)}')

    targets = []
    if args.target in ('pretrain', 'both'):
        targets.append(('pretrain', pretrain_index, args.out_dir / 'timeseries_pretrain.h5'))
    if args.target in ('exop_hosts', 'both'):
        targets.append(('exop_hosts', exop_index, args.out_dir / 'timeseries_exop_hosts.h5'))

    for name, index, h5_path in targets:
        print(f'\nResuming {name}: {h5_path}')
        if not h5_path.exists():
            raise FileNotFoundError(f'{h5_path} does not exist — run build_pretrain_h5.py first.')
        resume_write(h5_path, index, meta_lookup, host_meta_lookup)
        print(f'  {name} done.')

    print('\nDone.')


if __name__ == '__main__':
    main()
