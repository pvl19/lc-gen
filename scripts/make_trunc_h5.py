#!/usr/bin/env python3
"""
Create a truncated HDF5 file by copying the first `seq_len` timesteps from the
source HDF5. Useful for quick tests with shorter sequences.

Usage:
    python3 scripts/make_trunc_h5.py --input data/real_lightcurves/timeseries.h5 \
        --output tmp/trunc_2048.h5 --seq_len 2048
"""
import argparse
from pathlib import Path
import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seq_len', type=int, default=2048)
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {in_path}")

    with h5py.File(in_path, 'r') as fin:
        # Expect at least 'flux' and 'time'
        if 'flux' not in fin or 'time' not in fin:
            raise RuntimeError('Input HDF5 must contain datasets "flux" and "time"')
        flux = fin['flux'][:]
        time = fin['time'][:]
        flux_err = fin['flux_err'][:] if 'flux_err' in fin else None

    seq_len = args.seq_len
    if flux.shape[1] < seq_len:
        print(f"Warning: input seq_len={flux.shape[1]} < requested {seq_len}. Using original length.")
        seq_len = flux.shape[1]

    flux_trunc = flux[:, :seq_len]
    time_trunc = time[:, :seq_len]
    if flux_err is not None:
        flux_err_trunc = flux_err[:, :seq_len]
    else:
        flux_err_trunc = None

    with h5py.File(out_path, 'w') as fout:
        fout.create_dataset('flux', data=flux_trunc, compression='gzip')
        fout.create_dataset('time', data=time_trunc, compression='gzip')
        if flux_err_trunc is not None:
            fout.create_dataset('flux_err', data=flux_err_trunc, compression='gzip')

    print(f"Wrote truncated HDF5 to {out_path} with seq_len={seq_len}")


if __name__ == '__main__':
    main()
