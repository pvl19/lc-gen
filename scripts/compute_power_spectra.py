#!/usr/bin/env python3
"""Compute multi-taper power spectra, f-statistics, and ACFs for H5 light curves.

Reads flux/time/length from an HDF5 file, computes spectral features using
tapify's MultiTaper estimator, and writes power/f_stat/acf/freq to a separate
output H5 file.  These arrays are consumed by the conv encoder branch of
BiDirectionalMinGRU during pretraining.

Fixes over the original proj_lc_ages notebook:
  1. F-stat area-normalized over the frequency grid (consistent with power).
  2. ACF computed from the original multi-taper power spectrum *before*
     regridding/renormalization, then resampled to match the output grid.
"""

import argparse
import os
import sys
import time as timer

import h5py
import numpy as np
from tapify import MultiTaper


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_multitaper(flux, time, length, nw=4, k=7,
                       max_freq=50.0, freq_res=0.02):
    """Compute power spectrum, f-statistic, and ACF for one light curve.

    Returns
    -------
    final_freq  : (n_bins,) uniform frequency grid in cycles/day
    final_power : (n_bins,) area-normalized power spectrum
    final_fstat : (n_bins,) area-normalized f-statistic
    final_acf   : (n_bins,) ACF from original power, resampled to grid
    """
    valid_flux = flux[:length]
    valid_time = time[:length]

    mask = np.isfinite(valid_flux) & np.isfinite(valid_time)
    valid_flux = valid_flux[mask]
    valid_time = valid_time[mask]

    if len(valid_flux) < 9:
        n_bins = int(max_freq / freq_res)
        return (np.arange(0, max_freq, freq_res),
                np.zeros(n_bins, dtype=np.float32),
                np.zeros(n_bins, dtype=np.float32),
                np.zeros(n_bins, dtype=np.float32))

    mt = MultiTaper(x=valid_flux, t=valid_time, NW=nw, K=k)
    freq, power, fstat = mt.periodogram(
        method='fft',
        adaptive_weighting=True,
        N_padded='default',
        ftest=True,
    )

    # 1. ACF from ORIGINAL power spectrum (before regridding)
    acf_raw = np.fft.ifft(power).real
    if acf_raw[0] != 0:
        acf_raw /= acf_raw[0]

    # 2. Regrid power and f-stat to uniform frequency grid
    final_freq = np.arange(0, max_freq, freq_res)
    final_power = np.interp(final_freq, freq, power)
    final_fstat = np.interp(final_freq, freq, fstat)

    # 3. Area-normalize power and f-stat consistently
    power_area = np.trapz(final_power, final_freq)
    if power_area > 0:
        final_power /= power_area
    fstat_area = np.trapz(final_fstat, final_freq)
    if fstat_area > 0:
        final_fstat /= fstat_area

    # 4. Resample ACF to match grid length
    acf_x = np.linspace(0, 1, len(acf_raw))
    final_x = np.linspace(0, 1, len(final_freq))
    final_acf = np.interp(final_x, acf_x, acf_raw)

    return final_freq, final_power.astype(np.float32), \
           final_fstat.astype(np.float32), final_acf.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute multi-taper power spectra, f-statistics, and ACFs '
                    'for all light curves in an HDF5 file.')
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to HDF5 file with flux/time/length datasets')
    parser.add_argument('--output_h5', type=str, default=None,
                        help='Output H5 file for spectral data. Defaults to '
                             '<h5_path stem>_spectra.h5 alongside the input file.')
    parser.add_argument('--nw', type=float, default=4.0,
                        help='Time-bandwidth product for DPSS tapers (default: 4)')
    parser.add_argument('--k', type=int, default=7,
                        help='Number of Slepian tapers (default: 7)')
    parser.add_argument('--max_freq', type=float, default=50.0,
                        help='Maximum frequency in cycles/day (default: 50)')
    parser.add_argument('--freq_res', type=float, default=0.02,
                        help='Frequency grid resolution in cycles/day (default: 0.02)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Process only the first N samples (for debugging)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file')
    args = parser.parse_args()

    # Derive output path
    if args.output_h5 is None:
        stem = args.h5_path.rsplit('.h5', 1)[0]
        args.output_h5 = stem + '_spectra.h5'

    # Open source H5 (read-only) to get sample count
    with h5py.File(args.h5_path, 'r') as f:
        n_total = f['flux'].shape[0]

    if os.path.exists(args.output_h5) and not args.overwrite:
        print(f'Output file {args.output_h5} already exists.')
        print('Use --overwrite to replace it.')
        sys.exit(0)

    n_samples = min(n_total, args.max_samples) if args.max_samples else n_total
    n_bins = int(args.max_freq / args.freq_res)

    print(f'Computing power spectra for {n_samples} / {n_total} samples')
    print(f'  NW={args.nw}, K={args.k}')
    print(f'  Freq grid: 0–{args.max_freq} c/d, resolution {args.freq_res} c/d '
          f'→ {n_bins} bins')
    print(f'  Input:  {args.h5_path}')
    print(f'  Output: {args.output_h5}')

    # Write to a temp file first, then rename on success.
    # If the process crashes, the temp file is left behind but the output
    # file (and the source H5) are never in a corrupt state.
    tmp_path = args.output_h5 + '.tmp'

    # Create output H5 with pre-allocated datasets
    with h5py.File(tmp_path, 'w') as out:
        out.create_dataset('power', shape=(n_samples, n_bins), dtype=np.float32,
                           chunks=(1, n_bins), compression='gzip', compression_opts=4)
        out.create_dataset('f_stat', shape=(n_samples, n_bins), dtype=np.float32,
                           chunks=(1, n_bins), compression='gzip', compression_opts=4)
        out.create_dataset('acf', shape=(n_samples, n_bins), dtype=np.float32,
                           chunks=(1, n_bins), compression='gzip', compression_opts=4)

    n_failed = 0
    freq_grid = None
    t_start = timer.time()

    # Stream: read from source, compute, write to output
    with h5py.File(args.h5_path, 'r') as src, \
         h5py.File(tmp_path, 'a') as out:
        flux_ds = src['flux']
        time_ds = src['time']
        length_ds = src['length']
        power_ds = out['power']
        fstat_ds = out['f_stat']
        acf_ds = out['acf']

        for i in range(n_samples):
            if i % 500 == 0:
                elapsed = timer.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                print(f'  [{i:>6d}/{n_samples}]  '
                      f'{elapsed:.0f}s elapsed  ({rate:.1f} samples/s)')

            try:
                freq, power, fstat, acf = compute_multitaper(
                    flux=flux_ds[i],
                    time=time_ds[i],
                    length=int(length_ds[i]),
                    nw=args.nw,
                    k=args.k,
                    max_freq=args.max_freq,
                    freq_res=args.freq_res,
                )
                power_ds[i] = power
                fstat_ds[i] = fstat
                acf_ds[i] = acf
                if freq_grid is None:
                    freq_grid = freq
            except Exception as e:
                print(f'  WARNING: sample {i} failed: {e}')
                n_failed += 1

        # Write freq grid (small — 1D)
        if freq_grid is not None:
            out.create_dataset('freq', data=freq_grid, compression='gzip',
                               compression_opts=4)

    # Atomic rename: only replaces output file after everything succeeded
    if os.path.exists(args.output_h5):
        os.remove(args.output_h5)
    os.rename(tmp_path, args.output_h5)

    elapsed = timer.time() - t_start
    print(f'\nDone: {n_samples - n_failed} succeeded, {n_failed} failed '
          f'({elapsed:.0f}s total)')
    print(f'Written: {args.output_h5} — power/f_stat/acf ({n_samples}, {n_bins}), '
          f'freq ({n_bins},)')


if __name__ == '__main__':
    main()
