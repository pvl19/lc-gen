"""Convert exop_hosts/lc_data.pickle to an H5 file matching the format of data/timeseries_x.h5.

Only stores the fields required for age inference:
  flux, flux_err, time, length
  metadata/GaiaDR3_ID, metadata/tic, metadata/sector,
  metadata/BPRP0, metadata/e_BPRP0, metadata/Tmag

Usage:
    python scripts/convert_pickle_to_h5.py \
        --pickle   exop_hosts/lc_data.pickle \
        --meta_csv exop_hosts/host_all_metadata.csv \
        --output   exop_hosts/lc_data.h5
"""
import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

MAX_LEN = 16384   # matches data/timeseries_x.h5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle',   default='exop_hosts/lc_data.pickle')
    parser.add_argument('--meta_csv', default='exop_hosts/host_all_metadata.csv')
    parser.add_argument('--output',   default='exop_hosts/lc_data.h5')
    args = parser.parse_args()

    print(f'Loading pickle: {args.pickle}')
    with open(args.pickle, 'rb') as f:
        data = pickle.load(f)

    print(f'Loading metadata: {args.meta_csv}')
    meta_df = pd.read_csv(args.meta_csv)
    meta_df['GaiaDR3_ID'] = meta_df['GaiaDR3_ID'].astype(str)
    tic_to_gaia = dict(zip(meta_df['TIC_ID'], meta_df['GaiaDR3_ID']))

    # Filter to entries with a known Gaia ID
    entries = []
    skipped = 0
    for key, val in data.items():
        tic = int(val['TIC_ID'])
        if tic not in tic_to_gaia:
            skipped += 1
            continue
        entries.append((key, val, tic, tic_to_gaia[tic]))

    print(f'  {len(entries)} entries with Gaia IDs ({skipped} skipped — no Gaia match)')

    N = len(entries)
    flux_arr     = np.zeros((N, MAX_LEN), dtype=np.float32)
    flux_err_arr = np.zeros((N, MAX_LEN), dtype=np.float32)
    time_arr     = np.zeros((N, MAX_LEN), dtype=np.float32)
    length_arr   = np.zeros(N, dtype=np.int32)
    gaia_arr     = np.empty(N, dtype='S64')
    tic_arr      = np.zeros(N, dtype=np.int64)
    sector_arr   = np.zeros(N, dtype=np.int64)
    bprp0_arr    = np.full(N, np.nan, dtype=np.float64)
    bprp0_err_arr= np.full(N, np.nan, dtype=np.float64)
    tmag_arr     = np.full(N, np.nan, dtype=np.float64)

    for i, (key, val, tic, gaia_id) in enumerate(entries):
        flux     = np.asarray(val['flux'],     dtype=np.float32)
        flux_err = np.asarray(val['flux_err'], dtype=np.float32)
        time     = np.asarray(val['time'],     dtype=np.float32)

        L = min(len(flux), MAX_LEN)
        flux_arr[i, :L]     = flux[:L]
        flux_err_arr[i, :L] = flux_err[:L]
        time_arr[i, :L]     = time[:L]
        length_arr[i]        = L

        gaia_arr[i]      = gaia_id.encode('utf-8')
        tic_arr[i]       = tic
        sector_arr[i]    = int(val['sector'])

        m = val['metadata']
        bprp0_arr[i]     = float(m.get('BPRP0',     np.nan))
        bprp0_err_arr[i] = float(m.get('BPRP0_err', np.nan))
        tmag_arr[i]      = float(m.get('Tmag',      np.nan))

        if (i + 1) % 2000 == 0:
            print(f'  {i + 1}/{N}')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'\nWriting {out_path}')
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('flux',     data=flux_arr,     compression='gzip', compression_opts=4)
        f.create_dataset('flux_err', data=flux_err_arr, compression='gzip', compression_opts=4)
        f.create_dataset('time',     data=time_arr,     compression='gzip', compression_opts=4)
        f.create_dataset('length',   data=length_arr)

        grp = f.create_group('metadata')
        grp.create_dataset('GaiaDR3_ID', data=gaia_arr)
        grp.create_dataset('tic',        data=tic_arr)
        grp.create_dataset('sector',     data=sector_arr)
        grp.create_dataset('BPRP0',      data=bprp0_arr)
        grp.create_dataset('e_BPRP0',    data=bprp0_err_arr)
        grp.create_dataset('Tmag',       data=tmag_arr)

    print(f'Done. {N} light curves written.')
    print(f'  Unique TIC IDs: {len(np.unique(tic_arr))}')
    print(f'  Unique Gaia IDs: {len(np.unique(gaia_arr))}')


if __name__ == '__main__':
    main()
