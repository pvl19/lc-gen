#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute multitaper PSD, ACF, and F-statistics from light curves using tapify.

This script demonstrates how to:
1. Load irregularly sampled light curve data
2. Compute multitaper power spectral density (PSD)
3. Derive autocorrelation function (ACF) from PSD
4. Compute F-statistics for harmonic detection
5. Save the results for training models

Usage:
    python scripts/compute_spectra.py --input data/light_curves.pt --output data/spectra.pt --NW 4.0 --K 7

Examples:
    # Basic usage with default parameters
    python scripts/compute_spectra.py --input data/star_dataset.pt

    # Custom multitaper parameters
    python scripts/compute_spectra.py --input data/star_dataset.pt --NW 3.5 --K 6

    # Include F-statistics
    python scripts/compute_spectra.py --input data/star_dataset.pt --fstat
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lcgen.data import (
    compute_multitaper_psd,
    psd_to_acf,
    batch_compute_spectra,
    preprocess_lightcurve_for_spectral,
    prepare_spectra_for_model,
)


def load_light_curves(file_path: str) -> Dict[str, Any]:
    """
    Load light curve data from file.

    Supports .pt (PyTorch) and .pickle formats.

    Expected format:
    - Dictionary with keys: 'time', 'flux', 'flux_err' (all lists/arrays)
    - OR PyTorch dataset with __getitem__ returning (x, t, flux, flux_err)

    Returns
    -------
    data : dict
        Dictionary with 'times', 'fluxes', 'flux_errs' lists
    """
    file_path = Path(file_path)

    if file_path.suffix == '.pt':
        data_raw = torch.load(file_path)

        # Handle different formats
        if isinstance(data_raw, dict):
            if 'time' in data_raw:
                # Direct dictionary format
                return {
                    'times': data_raw['time'],
                    'fluxes': data_raw['flux'],
                    'flux_errs': data_raw.get('flux_err', None)
                }
            else:
                raise ValueError("Unknown data format in .pt file")

        elif hasattr(data_raw, '__getitem__'):
            # Dataset format - extract all samples
            times, fluxes, flux_errs = [], [], []
            for i in range(len(data_raw)):
                item = data_raw[i]
                if len(item) >= 4:
                    x, t, flux, flux_err = item[:4]
                    times.append(t.numpy() if torch.is_tensor(t) else t)
                    fluxes.append(flux.numpy() if torch.is_tensor(flux) else flux)
                    flux_errs.append(flux_err.numpy() if torch.is_tensor(flux_err) else flux_err)

            return {
                'times': times,
                'fluxes': fluxes,
                'flux_errs': flux_errs if flux_errs else None
            }

    elif file_path.suffix == '.pickle':
        import pickle
        with open(file_path, 'rb') as f:
            data_raw = pickle.load(f)

        return {
            'times': data_raw['time'],
            'fluxes': data_raw['flux'],
            'flux_errs': data_raw.get('flux_err', None)
        }

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute multitaper PSD, ACF, and F-statistics from light curves"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input light curve file (.pt or .pickle)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file for spectra (.pt or .pickle). If not specified, uses input_spectra.pt'
    )
    parser.add_argument(
        '--NW',
        type=float,
        default=4.0,
        help='Time-bandwidth parameter for multitaper (default: 4.0). Typical: 2.5-4.0'
    )
    parser.add_argument(
        '--K',
        type=int,
        default=None,
        help='Number of tapers. If not specified, defaults to 2*NW-1'
    )
    parser.add_argument(
        '--n-bins',
        type=int,
        default=1024,
        help='Number of frequency bins for output (default: 1024)'
    )
    parser.add_argument(
        '--fstat',
        action='store_true',
        help='Also compute F-statistics'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='nufft',
        choices=['nufft', 'fft'],
        help='Computation method (default: nufft - faster for irregular sampling)'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Preprocess light curves (remove outliers, etc.)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize spectra for model input (arcsinh transformation)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress'
    )

    args = parser.parse_args()

    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = input_path.parent / f"{input_path.stem}_spectra.pt"

    print("="*60)
    print("Multitaper Spectral Analysis")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Parameters: NW={args.NW}, K={args.K or f'2*NW-1={int(2*args.NW-1)}'}")
    print(f"Method: {args.method}")
    print(f"Frequency bins: {args.n_bins}")
    print(f"Compute F-statistics: {args.fstat}")
    print("="*60)

    # Load light curves
    print("\nLoading light curves...")
    try:
        data = load_light_curves(args.input)
        times = data['times']
        fluxes = data['fluxes']
        flux_errs = data['flux_errs']

        n_lcs = len(times)
        print(f"Loaded {n_lcs} light curves")

        # Print sample statistics
        if n_lcs > 0:
            sample_len = len(times[0])
            sample_duration = times[0][-1] - times[0][0] if len(times[0]) > 0 else 0
            print(f"Example: {sample_len} points, duration = {sample_duration:.2f} days")

    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Preprocess if requested
    if args.preprocess:
        print("\nPreprocessing light curves...")
        times_clean, fluxes_clean, flux_errs_clean = [], [], []

        for i in range(n_lcs):
            t_c, f_c, fe_c = preprocess_lightcurve_for_spectral(
                times[i],
                fluxes[i],
                flux_errs[i] if flux_errs is not None else None
            )
            times_clean.append(t_c)
            fluxes_clean.append(f_c)
            if fe_c is not None:
                flux_errs_clean.append(fe_c)

        times = times_clean
        fluxes = fluxes_clean
        if flux_errs_clean:
            flux_errs = flux_errs_clean
        print("Preprocessing complete")

    # Compute spectra
    print("\nComputing multitaper spectra...")
    try:
        results = batch_compute_spectra(
            times=times,
            fluxes=fluxes,
            flux_errs=flux_errs,
            NW=args.NW,
            K=args.K,
            n_bins=args.n_bins,
            return_fstat=args.fstat,
            verbose=args.verbose
        )

        print(f"✓ Computed {len(results['psd'])} PSDs")
        print(f"✓ Computed {len(results['acf'])} ACFs")
        if args.fstat:
            print(f"✓ Computed {len(results['fstat'])} F-statistics")

    except Exception as e:
        print(f"Error computing spectra: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Normalize if requested
    if args.normalize:
        print("\nNormalizing spectra for model input...")
        prepared = prepare_spectra_for_model(
            psd=results['psd'],
            acf=results['acf'],
            fstat=results.get('fstat', None),
            normalize_psd=True,
            normalize_acf=True
        )
        results['psd'] = prepared['psd']
        results['acf'] = prepared['acf']
        if 'fstat' in prepared:
            results['fstat'] = prepared['fstat']
        print("Normalization complete")

    # Save results
    print(f"\nSaving results to {args.output}...")
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.pt':
            torch.save(results, output_path)
        elif output_path.suffix == '.pickle':
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
        else:
            # Default to .pt
            output_path = output_path.with_suffix('.pt')
            torch.save(results, output_path)

        print(f"✓ Saved to {output_path}")

        # Print summary statistics
        print("\n" + "="*60)
        print("Summary Statistics")
        print("="*60)
        print(f"Number of spectra: {len(results['psd'])}")
        print(f"Frequency bins: {results['psd'].shape[1]}")
        print(f"Frequency range: {results['frequency'][0]:.6f} - {results['frequency'][-1]:.6f}")
        print(f"PSD shape: {results['psd'].shape}")
        print(f"ACF shape: {results['acf'].shape}")
        if 'fstat' in results:
            print(f"F-stat shape: {results['fstat'].shape}")
            print(f"Max F-statistic: {np.max(results['fstat']):.2f}")
        print("="*60)

        return 0

    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
