"""Convert mock light curve pickle to HDF5 format for Transformer training"""

import argparse
import pickle
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm


def convert_to_h5(input_pkl, output_h5, max_length=512):
    """
    Convert pickle file with light curves to HDF5 with fixed-length arrays.

    Args:
        input_pkl: Path to input pickle file
        output_h5: Path to output HDF5 file
        max_length: Maximum sequence length (will pad/truncate)
    """
    print(f"Loading light curves from {input_pkl}...")
    with open(input_pkl, 'rb') as f:
        lc_data = pickle.load(f)

    # Extract data - format is {'times': [...], 'fluxes': [...], 'flux_errs': [...], 'metadatas': [...]}
    times = lc_data['times']
    fluxes = lc_data['fluxes']

    n_samples = len(times)
    print(f"Found {n_samples} light curves")

    # Prepare arrays
    flux_data = np.zeros((n_samples, max_length), dtype=np.float32)
    time_data = np.zeros((n_samples, max_length), dtype=np.float32)
    lengths = np.zeros(n_samples, dtype=np.int32)

    print(f"Converting to fixed-length arrays (max_length={max_length})...")
    for i in tqdm(range(n_samples)):
        time = times[i]
        flux = fluxes[i]

        # Store actual length
        lengths[i] = len(time)

        if len(time) > max_length:
            # Truncate
            flux_data[i] = flux[:max_length]
            time_data[i] = time[:max_length]
        else:
            # Pad with zeros
            flux_data[i, :len(flux)] = flux
            time_data[i, :len(time)] = time
            # Note: padding positions will be zeros

    # Save to HDF5
    print(f"Saving to {output_h5}...")
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('flux', data=flux_data, compression='gzip')
        f.create_dataset('time', data=time_data, compression='gzip')
        f.create_dataset('length', data=lengths, compression='gzip')

        # Metadata
        f.attrs['n_samples'] = n_samples
        f.attrs['max_length'] = max_length
        f.attrs['description'] = 'Light curve time-series for Transformer training'

    print(f"Done! Saved {n_samples} light curves to {output_h5}")
    print(f"  flux: {flux_data.shape}")
    print(f"  time: {time_data.shape}")
    print(f"  Average length: {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"  Truncated: {(lengths > max_length).sum()} ({100*(lengths > max_length).sum()/n_samples:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Convert light curves to HDF5 for Transformer')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input pickle file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output HDF5 file')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    convert_to_h5(args.input, args.output, args.max_length)


if __name__ == '__main__':
    main()
