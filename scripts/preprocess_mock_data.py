"""
Preprocess mock light curves into multi-modal features (PSD/ACF/F-stat).

Loads mock light curves and runs batch preprocessing to generate:
- Power Spectral Density (1024 bins, arcsinh normalized)
- Autocorrelation Function (1024 bins, arcsinh normalized)
- F-statistic (1024 bins, z-score normalized)

Saves to HDF5 for efficient loading during training.
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import h5py
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lcgen.data import batch_preprocess_multimodal


def preprocess_dataset(input_file, output_file, batch_size=100, NW=4.0, K=7):
    """
    Preprocess all light curves in batches.

    Parameters
    ----------
    input_file : Path
        Input pickle file with mock light curves
    output_file : Path
        Output HDF5 file for preprocessed features
    batch_size : int
        Number of light curves to process at once
    NW : float
        Time-bandwidth parameter for multitaper
    K : int
        Number of tapers
    """
    print(f"Loading light curves from {input_file}...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    times = data['times']
    fluxes = data['fluxes']
    flux_errs = data['flux_errs']
    metadatas = data['metadatas']

    n_samples = len(times)
    print(f"Loaded {n_samples} light curves")

    print(f"\nPreprocessing with:")
    print(f"  NW = {NW}")
    print(f"  K = {K}")
    print(f"  Batch size = {batch_size}")

    # Process in batches
    all_acf = []
    all_psd = []
    all_fstat = []
    all_pvalues = []
    all_tabular = []
    all_frequency = None  # Same for all
    failed_indices = []

    n_batches = int(np.ceil(n_samples / batch_size))

    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)

        batch_times = times[start_idx:end_idx]
        batch_fluxes = fluxes[start_idx:end_idx]
        batch_flux_errs = flux_errs[start_idx:end_idx]

        try:
            # Preprocess batch
            result = batch_preprocess_multimodal(
                batch_times,
                batch_fluxes,
                batch_flux_errs,
                NW=NW,
                K=K,
                n_bins=1024,
                freq_spacing='log',
                compute_pvalues=True,
                verbose=False
            )

            # Store results
            all_acf.append(result['acf'])
            all_psd.append(result['psd'])
            all_fstat.append(result['fstat'])
            all_pvalues.append(result['pvalues'])
            all_tabular.append(result['tabular'])

            if all_frequency is None:
                all_frequency = result['frequency']

        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")
            # Mark failed light curves
            for i in range(start_idx, end_idx):
                failed_indices.append(i)

    # Concatenate all batches
    print("\nConcatenating results...")
    acf_array = np.concatenate(all_acf, axis=0)
    psd_array = np.concatenate(all_psd, axis=0)
    fstat_array = np.concatenate(all_fstat, axis=0)
    pvalues_array = np.concatenate(all_pvalues, axis=0)
    tabular_array = np.concatenate(all_tabular, axis=0)

    print(f"\nPreprocessed shapes:")
    print(f"  ACF: {acf_array.shape}")
    print(f"  PSD: {psd_array.shape}")
    print(f"  F-stat: {fstat_array.shape}")
    print(f"  P-values: {pvalues_array.shape}")
    print(f"  Tabular: {tabular_array.shape}")
    print(f"  Frequency: {all_frequency.shape}")

    if failed_indices:
        print(f"\nWarning: {len(failed_indices)} light curves failed preprocessing")
        print(f"Failed indices: {failed_indices[:10]}..." if len(failed_indices) > 10 else failed_indices)

    # Save to HDF5
    print(f"\nSaving to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, 'w') as f:
        # Preprocessed features
        f.create_dataset('acf', data=acf_array, compression='gzip')
        f.create_dataset('psd', data=psd_array, compression='gzip')
        f.create_dataset('fstat', data=fstat_array, compression='gzip')
        f.create_dataset('pvalues', data=pvalues_array, compression='gzip')
        f.create_dataset('tabular', data=tabular_array, compression='gzip')
        f.create_dataset('frequency', data=all_frequency, compression='gzip')

        # Metadata
        f.attrs['n_samples'] = len(acf_array)
        f.attrs['n_bins'] = 1024
        f.attrs['NW'] = NW
        f.attrs['K'] = K
        f.attrs['n_failed'] = len(failed_indices)

        # Save light curve types and sampling strategies
        lc_types = np.array([m['lc_type'] for m in metadatas], dtype='S20')
        sampling_strategies = np.array([m['sampling_strategy'] for m in metadatas], dtype='S20')

        # Remove failed indices
        if failed_indices:
            mask = np.ones(n_samples, dtype=bool)
            mask[failed_indices] = False
            lc_types = lc_types[mask]
            sampling_strategies = sampling_strategies[mask]

        f.create_dataset('lc_type', data=lc_types)
        f.create_dataset('sampling_strategy', data=sampling_strategies)

    print("Done!")
    print(f"\nSummary:")
    print(f"  Successfully preprocessed: {len(acf_array)} / {n_samples}")
    print(f"  Output file size: {output_file.stat().st_size / 1024**2:.1f} MB")


def verify_preprocessing(output_file):
    """Quick verification of preprocessed data."""
    print(f"\nVerifying {output_file}...")

    with h5py.File(output_file, 'r') as f:
        print(f"\nDatasets:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"  {key}: {f[key].shape}, dtype={f[key].dtype}")

        print(f"\nAttributes:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")

        # Check for NaNs
        acf = f['acf'][:]
        psd = f['psd'][:]
        fstat = f['fstat'][:]

        print(f"\nData quality:")
        print(f"  ACF NaNs: {np.sum(np.isnan(acf))}")
        print(f"  PSD NaNs: {np.sum(np.isnan(psd))}")
        print(f"  F-stat NaNs: {np.sum(np.isnan(fstat))}")

        print(f"\nValue ranges:")
        print(f"  ACF: [{np.min(acf):.3f}, {np.max(acf):.3f}]")
        print(f"  PSD: [{np.min(psd):.3f}, {np.max(psd):.3f}]")
        print(f"  F-stat: [{np.min(fstat):.3f}, {np.max(fstat):.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess mock light curves to multi-modal features"
    )
    parser.add_argument(
        '--input', type=str,
        default='data/mock_lightcurves/mock_lightcurves.pkl',
        help='Input pickle file with mock light curves'
    )
    parser.add_argument(
        '--output', type=str,
        default='data/mock_lightcurves/preprocessed.h5',
        help='Output HDF5 file for preprocessed features'
    )
    parser.add_argument(
        '--batch_size', type=int, default=100,
        help='Batch size for preprocessing (default: 100)'
    )
    parser.add_argument(
        '--NW', type=float, default=4.0,
        help='Time-bandwidth parameter (default: 4.0)'
    )
    parser.add_argument(
        '--K', type=int, default=7,
        help='Number of tapers (default: 7)'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify preprocessed data after creation'
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    preprocess_dataset(
        input_file,
        output_file,
        batch_size=args.batch_size,
        NW=args.NW,
        K=args.K
    )

    if args.verify:
        verify_preprocessing(output_file)
