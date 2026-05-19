"""Build a pretrain HDF5 file for the thick-disk sample.

Input:
  final_pretrain/lc_data_thickdisk.pickle  -- light curves, z-score (mean/std)
                                              normalized, keyed '{TIC}_{sector}'.
  final_pretrain/thickdisk_metadata.csv    -- per-star astrophysical metadata.

Output:
  final_pretrain/timeseries_thickdisk.h5   -- same format as timeseries_pretrain.h5.

The pickle flux is mean/std normalized. This script applies the SAME robust
re-normalization as the main pretrain pipeline — `renormalize_record` from
robust_renormalize_pickles.py — so the thick-disk flux is median-centred and
(p84-p16)/2-scaled, consistent with timeseries_pretrain.h5. It is done in a
single streaming pass (one 2.2 GB pickle held in memory, H5 written
incrementally) to keep the peak memory footprint near the pickle size.

Only light curves whose TIC_ID is present in thickdisk_metadata.csv are kept.
All rows get exop_host = 0 (thick-disk stars are pretrain, not exoplanet hosts).

Usage:
    python scripts/build_thickdisk_h5.py
    python scripts/build_thickdisk_h5.py --limit 64   # smoke test -> tiny h5
"""
import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_renormalize_pickles import renormalize_record

DATA_DIR = Path('final_pretrain')
# Mirror build_pretrain_h5.py exactly.
STR_FIELDS = ['GaiaDR3_ID']
NUM_FIELDS_CSV = ['Tmag', 'parallax', 'parallax_error', 'G0', 'G0_err', 'BPRP0', 'BPRP0_err']
NUM_FIELDS_PICKLE = ['camera', 'ccd', 'mean_flux', 'std_flux', 'median_flux', 'iqr_half_flux']
CADENCE_S = 120.0


def create_h5(out_path, N, max_len):
    """Create the h5 file with pre-allocated datasets (matches build_pretrain_h5.create_h5)."""
    f = h5py.File(out_path, 'w')
    f.attrs['n_samples'] = N
    f.attrs['max_length'] = max_len
    # One row per chunk: each row is written exactly once, so HDF5 compresses
    # it once with no decompress/recompress cycles (a multi-row chunk would be
    # recompressed on every row write). This is also the optimal layout for
    # LazyH5Dataset, which reads one row at a time. Storage detail only —
    # dataset shapes/dtypes/values are identical to the other H5 files.
    f.create_dataset('flux',     shape=(N, max_len), dtype=np.float32, chunks=(1, max_len), compression='gzip', compression_opts=4)
    f.create_dataset('flux_err', shape=(N, max_len), dtype=np.float32, chunks=(1, max_len), compression='gzip', compression_opts=4)
    f.create_dataset('time',     shape=(N, max_len), dtype=np.float64, chunks=(1, max_len), compression='gzip', compression_opts=4)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pickle', type=Path, default=DATA_DIR / 'lc_data_thickdisk.pickle')
    ap.add_argument('--meta_csv', type=Path, default=DATA_DIR / 'thickdisk_metadata.csv')
    ap.add_argument('--out', type=Path, default=DATA_DIR / 'timeseries_thickdisk.h5')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process only the first N matched light curves (smoke test).')
    args = ap.parse_args()

    print('Loading metadata...')
    # GaiaDR3_ID is a 19-digit integer — read it as str so it is not coerced to
    # float64 (which loses precision and serialises as scientific notation).
    meta_df = pd.read_csv(args.meta_csv, dtype={'GaiaDR3_ID': str})
    meta_df['TIC_ID'] = meta_df['TIC_ID'].astype(np.int64)
    meta_lookup = {int(r['TIC_ID']): r for _, r in meta_df.iterrows()}
    print(f'  {len(meta_lookup)} unique TICs in {args.meta_csv.name}')

    print(f'Loading {args.pickle} (this is the large step)...')
    with open(args.pickle, 'rb') as f:
        data = pickle.load(f)
    print(f'  {len(data)} light curves in pickle')

    # Pass 1: select records whose TIC is in the CSV; record max length.
    keys, max_len, skipped = [], 0, 0
    for key, lc in data.items():
        if int(lc['TIC_ID']) not in meta_lookup:
            skipped += 1
            continue
        keys.append(key)
        max_len = max(max_len, len(lc['flux']))
    keys.sort()
    if args.limit is not None:
        keys = keys[:args.limit]
        max_len = max(len(data[k]['flux']) for k in keys)
    N = len(keys)
    print(f'  matched {N} light curves, skipped {skipped} (TIC not in CSV); max_len={max_len}')
    if N == 0:
        raise SystemExit('No matched light curves — nothing to write.')

    # Pass 2: renormalize + write, one record at a time.
    print(f'Writing {args.out} ...')
    h5 = create_h5(args.out, N, max_len)
    renorm_skipped = 0
    written = 0
    for row_i, key in enumerate(keys):
        rec = renormalize_record(data[key])      # mean/std -> robust median/iqr
        flux     = np.asarray(rec['flux'],     dtype=np.float32)
        flux_err = np.asarray(rec['flux_err'], dtype=np.float32)
        time     = np.asarray(rec['time'],     dtype=np.float64)
        time = time - time[0]                    # reset to sector start

        # Replace NaN/missing flux_err with the light curve's median (as in
        # build_pretrain_h5.write_entries).
        bad = ~np.isfinite(flux_err)
        if bad.any():
            med = np.nanmedian(flux_err)
            flux_err[bad] = med if np.isfinite(med) else 0.0

        L = len(flux)
        h5['flux'][row_i, :L]     = flux
        h5['flux_err'][row_i, :L] = flux_err
        h5['time'][row_i, :L]     = time
        h5['length'][row_i]       = L
        tic = int(rec['TIC_ID'])
        h5['metadata/tic'][row_i]       = tic
        h5['metadata/sector'][row_i]    = int(rec['sector'])
        h5['metadata/exop_host'][row_i] = 0      # thick-disk -> pretrain set

        meta_row = meta_lookup[tic]
        for field in STR_FIELDS:
            val = meta_row.get(field, '')
            h5[f'metadata/{field}'][row_i] = str(val).encode('utf-8') if pd.notna(val) else b''
        for field in NUM_FIELDS_CSV:
            val = meta_row.get(field, np.nan)
            h5[f'metadata/{field}'][row_i] = float(val) if pd.notna(val) else np.nan
        for field in NUM_FIELDS_PICKLE:
            val = rec['metadata'].get(field, np.nan)
            try:
                h5[f'metadata/{field}'][row_i] = float(val)
            except (TypeError, ValueError):
                h5[f'metadata/{field}'][row_i] = np.nan

        written += 1
        if written % 1000 == 0:
            print(f'  {written}/{N}')
    h5.close()

    with h5py.File(args.out, 'r') as f:
        n = int(f.attrs['n_samples'])
        tics = len(np.unique(f['metadata/tic'][:]))
    print(f'Done. {args.out.name}: {n} light curves, {tics} unique TICs.')


if __name__ == '__main__':
    main()
