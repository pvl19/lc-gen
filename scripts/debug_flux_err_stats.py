#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'src'))
from lcgen.models.TimeSeriesDataset import TimeSeriesDataset, collate_fn
from lcgen.utils.loss import recon_loss


def stats(arr):
    arr = np.asarray(arr)
    return {
        'min': float(np.nanmin(arr)),
        '1%': float(np.nanpercentile(arr, 1)),
        '5%': float(np.nanpercentile(arr, 5)),
        '25%': float(np.nanpercentile(arr, 25)),
        '50%': float(np.nanpercentile(arr, 50)),
        '75%': float(np.nanpercentile(arr, 75)),
        '95%': float(np.nanpercentile(arr, 95)),
        '99%': float(np.nanpercentile(arr, 99)),
        'max': float(np.nanmax(arr)),
        'nan_count': int(np.isnan(arr).sum()),
        'inf_count': int(np.isinf(arr).sum()),
        'zero_count': int((arr == 0).sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/timeseries.h5')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--random_seed', type=int, default=19)
    p.add_argument('--mock_sinusoid', action='store_true')
    args = p.parse_args()

    ds = TimeSeriesDataset(args.input, args.random_seed, 2, 10, 0.2, max_length=256, num_samples=None, mock_sinusoid=args.mock_sinusoid)
    print('Loaded dataset shapes: flux', ds.flux.shape, 'flux_err', ds.flux_err.shape, 'time', ds.time.shape)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    flux, flux_err, times = batch
    print('\nflux stats:')
    print(stats(flux.numpy()))
    print('\nflux_err stats:')
    ferr_stats = stats(flux_err.numpy())
    for k, v in ferr_stats.items():
        print(f'  {k}: {v}')

    # Count very small errors
    ferr = flux_err.numpy()
    small_mask = (ferr < 1e-3)
    print('\nCounts:')
    print('  total elements:', ferr.size)
    print('  <1e-3:', int(small_mask.sum()))
    print('  <1e-6:', int((ferr < 1e-6).sum()))

    # Compute recon_loss with a dummy prediction (mean prediction 0)
    recon = torch.zeros_like(flux)
    nll = recon_loss(flux, flux_err, recon)  # (B, L)
    nll_np = nll.numpy()
    print('\nrecon nll stats:')
    print(stats(nll_np))
    print('max nll:', np.nanmax(nll_np))
    max_idx = np.unravel_index(np.nanargmax(nll_np), nll_np.shape)
    print('max nll at index', max_idx)
    print('  flux at idx:', float(flux.numpy()[max_idx]))
    print('  flux_err at idx:', float(ferr[max_idx]))
    print('  recon at idx:', float(recon.numpy()[max_idx]))
    print('  var at idx:', float((abs(ferr[max_idx]) if not np.isnan(ferr[max_idx]) else 0.0)**2))

if __name__ == '__main__':
    main()
