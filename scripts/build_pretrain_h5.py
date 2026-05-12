"""Build pretrain HDF5 files from lc_data_p1-p7 pickle files and metadata.csv.

Produces two files:
  final_pretrain/timeseries_pretrain.h5      -- exop_host = 0
  final_pretrain/timeseries_exop_hosts.h5   -- exop_host = 1

Only TIC IDs present in both the pickle files and metadata.csv are included.
flux_err NaN/missing values are replaced with the median flux_err of that light curve.
All light curve points are stored (no truncation). Arrays are zero-padded to the
longest light curve in each output file; the 'length' dataset records the true length.

Usage:
    python scripts/build_pretrain_h5.py
"""

import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DATA_DIR = Path('final_pretrain')
DEFAULT_PICKLE_FILES = [DATA_DIR / f'lc_data_p{i}.pickle' for i in range(1, 8)]
DEFAULT_META_CSV = DATA_DIR / 'metadata.csv'
DEFAULT_HOST_META_CSV = DATA_DIR / 'host_all_metadata.csv'
DEFAULT_OUT_PRETRAIN = DATA_DIR / 'timeseries_pretrain.h5'
DEFAULT_OUT_EXOP = DATA_DIR / 'timeseries_exop_hosts.h5'

# String metadata fields (stored as fixed-length bytes)
STR_FIELDS = ['GaiaDR3_ID']
# Numeric metadata fields from CSV
NUM_FIELDS_CSV = ['Tmag', 'parallax', 'parallax_error', 'G0', 'G0_err', 'BPRP0', 'BPRP0_err']
# Numeric metadata fields from pickle. mean_flux / std_flux describe the
# original (outlier-sensitive) z-score normalization; median_flux /
# iqr_half_flux describe the robust re-normalization written by
# scripts/robust_renormalize_pickles.py. Old pickles without the robust fields
# write NaN — downstream code can fall back to mean/std.
NUM_FIELDS_PICKLE = ['camera', 'ccd', 'mean_flux', 'std_flux',
                     'median_flux', 'iqr_half_flux']
# Constant metadata fields
CADENCE_S = 120.0


def scan_pickles(paths, meta_lookup, host_meta_lookup):
    """
    Pass 1: scan all pickles without retaining light curve data.
    Returns index info needed to pre-allocate h5 files:
      pretrain_index, exop_index — lists of (pickle_path, key, max_flux_len)
    """
    pretrain_index, exop_index = [], []
    skipped = 0
    for path in paths:
        print(f'  Scanning {path}...')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for key, lc in data.items():
            tic = int(lc['TIC_ID'])
            is_exop = lc['metadata']['exop_host'] == 1
            if tic in meta_lookup:
                pass
            elif is_exop and tic in host_meta_lookup:
                pass
            else:
                skipped += 1
                continue
            entry = (path, key, len(lc['flux']))
            if is_exop:
                exop_index.append(entry)
            else:
                pretrain_index.append(entry)
        del data
    print(f'  Skipped (no metadata): {skipped}')
    return pretrain_index, exop_index


def create_h5(out_path, N, max_len):
    """Create an h5 file with pre-allocated resizable datasets."""
    f = h5py.File(out_path, 'w')
    f.attrs['n_samples'] = N
    f.attrs['max_length'] = max_len

    chunk = min(256, N)
    f.create_dataset('flux',     shape=(N, max_len), dtype=np.float32, chunks=(chunk, max_len), compression='gzip', compression_opts=4)
    f.create_dataset('flux_err', shape=(N, max_len), dtype=np.float32, chunks=(chunk, max_len), compression='gzip', compression_opts=4)
    f.create_dataset('time',     shape=(N, max_len), dtype=np.float64, chunks=(chunk, max_len), compression='gzip', compression_opts=4)
    f.create_dataset('length',   shape=(N,), dtype=np.int32)

    grp = f.create_group('metadata')
    grp.create_dataset('tic',       shape=(N,), dtype=np.int64)
    grp.create_dataset('sector',    shape=(N,), dtype=np.int64)
    grp.create_dataset('exop_host', shape=(N,), dtype=np.int8)
    for field in STR_FIELDS:
        grp.create_dataset(field, shape=(N,), dtype='S64')
    for field in NUM_FIELDS_CSV:
        grp.create_dataset(field, shape=(N,), dtype=np.float64, fillvalue=np.nan)
    for field in NUM_FIELDS_PICKLE:
        grp.create_dataset(field, shape=(N,), dtype=np.float32)
    grp.create_dataset('cadence_s', data=np.full(N, CADENCE_S, dtype=np.float32))
    return f


def write_entries(h5_file, index, meta_lookup, host_meta_lookup):
    """
    Pass 2: load one pickle at a time and write matching entries into the open h5 file.
    index is a list of (pickle_path, key, flux_len) in the order they should be written.
    """
    # Group index rows by pickle file to load each file once
    from collections import defaultdict
    by_file = defaultdict(list)
    for row_i, (path, key, _) in enumerate(index):
        by_file[path].append((row_i, key))

    total = len(index)
    written = 0
    for path, rows in by_file.items():
        print(f'  Writing from {path} ({len(rows)} entries)...')
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

            # Reset time to sector start
            time = time - time[0]

            # Replace NaN/missing flux_err with median
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
            if written % 5000 == 0:
                print(f'  {written}/{total}')

        del data


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pickle_dir', type=Path, default=None,
                    help='Directory containing lc_data_p1..p7.pickle. '
                         'Default: final_pretrain/. Use final_pretrain_robust/ '
                         'after running robust_renormalize_pickles.py.')
    ap.add_argument('--meta_dir', type=Path, default=DATA_DIR,
                    help='Directory containing metadata.csv + host_all_metadata.csv. '
                         'Default: final_pretrain/.')
    ap.add_argument('--out_dir', type=Path, default=None,
                    help='Directory to write timeseries_*.h5 into. Default: '
                         'same as --pickle_dir (or final_pretrain/ if neither set).')
    args = ap.parse_args()

    pickle_dir = args.pickle_dir or DATA_DIR
    out_dir    = args.out_dir or pickle_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle_files = [pickle_dir / f'lc_data_p{i}.pickle' for i in range(1, 8)]
    meta_csv      = args.meta_dir / 'metadata.csv'
    host_meta_csv = args.meta_dir / 'host_all_metadata.csv'
    out_pretrain  = out_dir / 'timeseries_pretrain.h5'
    out_exop      = out_dir / 'timeseries_exop_hosts.h5'

    print(f'pickle_dir: {pickle_dir}')
    print(f'meta_dir:   {args.meta_dir}')
    print(f'out_dir:    {out_dir}')

    print('Loading metadata...')
    meta_df = pd.read_csv(meta_csv)
    meta_df['TIC_ID'] = meta_df['TIC_ID'].astype(int)
    meta_lookup = {row['TIC_ID']: row for _, row in meta_df.iterrows()}
    print(f'  {len(meta_lookup)} unique TICs in metadata.csv')

    host_meta_df = pd.read_csv(host_meta_csv)
    host_meta_df['TIC_ID'] = host_meta_df['TIC_ID'].astype(int)
    host_meta_lookup = {row['TIC_ID']: row for _, row in host_meta_df.iterrows()}
    print(f'  {len(host_meta_lookup)} unique TICs in host_all_metadata.csv')

    print('\nPass 1: scanning pickle files...')
    pretrain_index, exop_index = scan_pickles(pickle_files, meta_lookup, host_meta_lookup)
    print(f'  exop_host=0 (pretrain): {len(pretrain_index)}')
    print(f'  exop_host=1 (exop):     {len(exop_index)}')

    for index, out_path in [(pretrain_index, out_pretrain), (exop_index, out_exop)]:
        N = len(index)
        max_len = max(fl for _, _, fl in index)
        print(f'\nPass 2: writing {out_path} ({N} entries, max_len={max_len})...')
        h5_file = create_h5(out_path, N, max_len)
        write_entries(h5_file, index, meta_lookup, host_meta_lookup)
        h5_file.close()
        print(f'  Done.')

    print('\nSummary:')
    for path in (out_pretrain, out_exop):
        with h5py.File(path, 'r') as f:
            n = f.attrs['n_samples']
            tics = len(np.unique(f['metadata/tic'][:]))
        print(f'  {path.name}: {n} light curves, {tics} unique TICs')


if __name__ == '__main__':
    main()
