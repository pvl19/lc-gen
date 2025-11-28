#!/usr/bin/env python3
"""
Convert `star_sector_lc_formatted.pickle` into a compact HDF5 timeseries file.

This script expects the pickle to be a dict-like object with keys
`times`, `fluxes`, and `flux_errs` each mapping to a sequence (list) of
per-example arrays/lists. It pads or truncates each sequence to `seq_len`
and writes datasets `flux`, `time`, `flux_err`, and `length` into the
output HDF5 file using gzip compression.

Usage:
    python3 scripts/pickle_to_h5.py --input data/real_lightcurves/star_sector_lc_formatted.pickle \
        --output data/real_lightcurves/timeseries.h5 --seq_len 16384
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
import h5py


def pad_or_trunc(arr, target_len, pad_value=0.0, pad_mode='constant'):
    # arr: 1D numpy array
    n = len(arr)
    if n >= target_len:
        return np.asarray(arr[:target_len], dtype=float)
    out = np.empty((target_len,), dtype=float)
    out[:n] = np.asarray(arr, dtype=float)
    if pad_mode == 'constant':
        out[n:] = pad_value
    elif pad_mode == 'repeat_last':
        if n > 0:
            out[n:] = float(arr[-1])
        else:
            out[n:] = pad_value
    else:
        raise ValueError('unsupported pad_mode')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seq_len', type=int, default=16384)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)

    if not inp.exists():
        raise FileNotFoundError(f'Input pickle not found: {inp}')

    if outp.exists() and not args.overwrite:
        raise RuntimeError(f'Output HDF5 already exists: {outp} (use --overwrite to replace)')

    print(f'Loading pickle from {inp}...')
    with open(inp, 'rb') as f:
        data = pickle.load(f)

    # Expect dict-like structure with required keys
    if not isinstance(data, dict):
        raise RuntimeError('Expected pickle to contain a dict with keys: times, fluxes, flux_errs')

    for key in ('times', 'fluxes', 'flux_errs'):
        if key not in data:
            raise RuntimeError(f"Pickle missing required key: '{key}'")

    times_list = data['times']
    flux_list = data['fluxes']
    flux_errs_list = data['flux_errs']

    n = len(flux_list)
    if len(times_list) != n or len(flux_errs_list) != n:
        raise RuntimeError('Mismatched lengths between times/fluxes/flux_errs in pickle')

    seq_len = args.seq_len
    print(f'Converting {n} examples to fixed length {seq_len}...')

    flux_arr = np.zeros((n, seq_len), dtype=float)
    time_arr = np.zeros((n, seq_len), dtype=float)
    ferr_arr = np.zeros((n, seq_len), dtype=float)
    lengths = np.zeros((n,), dtype=np.int32)

    # compute a global fallback for flux_err padding if needed
    all_ferr_vals = []
    for fe in flux_errs_list:
        try:
            a = np.asarray(fe, dtype=float)
            if a.size > 0:
                all_ferr_vals.append(np.median(a))
        except Exception:
            continue
    if len(all_ferr_vals) > 0:
        global_ferr_median = float(np.median(all_ferr_vals))
    else:
        global_ferr_median = 1e-3

    for i in range(n):
        flux_i = np.asarray(flux_list[i], dtype=float)
        time_i = np.asarray(times_list[i], dtype=float)
        ferr_i = np.asarray(flux_errs_list[i], dtype=float)

        L = min(len(flux_i), seq_len)
        lengths[i] = L

        if len(flux_i) >= seq_len:
            flux_arr[i] = flux_i[:seq_len]
        else:
            flux_arr[i] = pad_or_trunc(flux_i, seq_len, pad_value=0.0, pad_mode='constant')

        if len(time_i) >= seq_len:
            time_arr[i] = time_i[:seq_len]
        else:
            # repeat last timestamp for padding if available, else zero
            pad_val = float(time_i[-1]) if len(time_i) > 0 else 0.0
            time_arr[i] = pad_or_trunc(time_i, seq_len, pad_value=pad_val, pad_mode='repeat_last')

        if len(ferr_i) >= seq_len:
            ferr_arr[i] = ferr_i[:seq_len]
        else:
            # pad flux_err with the per-sample median if available, otherwise global median
            pad_val = float(np.median(ferr_i)) if ferr_i.size > 0 else global_ferr_median
            # ensure non-zero small floor
            pad_val = max(pad_val, 1e-9)
            ferr_arr[i] = pad_or_trunc(ferr_i, seq_len, pad_value=pad_val, pad_mode='constant')

    # Write HDF5
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        outp.unlink()

    print(f'Writing HDF5 to {outp}...')
    with h5py.File(outp, 'w') as fout:
        fout.create_dataset('flux', data=flux_arr, compression='gzip')
        fout.create_dataset('time', data=time_arr, compression='gzip')
        fout.create_dataset('flux_err', data=ferr_arr, compression='gzip')
        fout.create_dataset('length', data=lengths, compression='gzip')

    print('Done.')


if __name__ == '__main__':
    main()
