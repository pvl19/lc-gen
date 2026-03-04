#!/usr/bin/env python
"""
Add metadata fields from TESS_CF_metadata.csv to pickle and h5 files.
Matches records by TIC_ID.
"""

import pickle
import h5py
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    # Paths
    csv_path = Path("data/TESS_CF_metadata.csv")
    pickle_path = Path("data/star_sector_lc_formatted_with_extra_channels.pickle")
    h5_path = Path("data/timeseries_x.h5")

    # Load CSV and create lookup by (TIC_ID, sector)
    print(f"Loading CSV from {csv_path}...")
    csv_df = pd.read_csv(csv_path)
    print(f"  CSV shape: {csv_df.shape}")

    # Create lookup dict by (TIC_ID, sector) tuple
    csv_lookup = {}
    for _, row in csv_df.iterrows():
        key = (int(row['TIC_ID']), int(row['sector']))
        csv_lookup[key] = row
    print(f"  Created lookup with {len(csv_lookup)} entries")

    # Define new columns to add (exclude TIC_ID and sector which already exist)
    existing_keys = ['tic', 'sector', 'duration', 'med_flux_error', 'n_points', 'mean_flux', 'std_flux',
                     'norm_G0', 'norm_G0_err', 'norm_BP0', 'norm_BP0_err', 'norm_RP0', 'norm_RP0_err',
                     'norm_parallax', 'norm_parallax_err', 'norm_log_mean_flux']
    new_columns = [c for c in csv_df.columns if c not in existing_keys and c != 'TIC_ID']
    print(f"  New columns to add: {new_columns}")

    # ========== Update pickle file ==========
    print(f"\nUpdating pickle file: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    n_records = len(data['metadatas'])
    n_matched = 0
    n_unmatched = 0

    for i, meta in enumerate(data['metadatas']):
        tic = int(meta['tic'])
        sector = int(meta['sector'])
        key = (tic, sector)

        if key in csv_lookup:
            row = csv_lookup[key]
            for col in new_columns:
                value = row[col]
                # Handle NaN values
                if pd.isna(value):
                    if col in ['GaiaDR3_ID', 'cluster', 'subgroup']:
                        meta[col] = ''
                    else:
                        meta[col] = np.nan
                else:
                    meta[col] = value
            n_matched += 1
        else:
            # Add NaN/empty values for unmatched records
            for col in new_columns:
                if col in ['GaiaDR3_ID', 'cluster', 'subgroup']:
                    meta[col] = ''
                else:
                    meta[col] = np.nan
            n_unmatched += 1

    print(f"  Matched: {n_matched}/{n_records}")
    print(f"  Unmatched: {n_unmatched}/{n_records}")

    # Save updated pickle (backup first)
    backup_pickle = pickle_path.with_suffix('.pickle.bak')
    print(f"  Creating backup: {backup_pickle}")
    import shutil
    shutil.copy(pickle_path, backup_pickle)

    print(f"  Saving updated pickle...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Done! New metadata keys: {list(data['metadatas'][0].keys())}")

    # ========== Update h5 file ==========
    print(f"\nUpdating h5 file: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        tic_ids = f['metadata']['tic'][:]
        sectors = f['metadata']['sector'][:]
        n_h5_records = len(tic_ids)

    # Create arrays for new fields
    new_data = {}
    for col in new_columns:
        if col in ['GaiaDR3_ID', 'cluster', 'subgroup']:
            # String columns
            new_data[col] = np.array([''] * n_h5_records, dtype='S64')
        else:
            # Numeric columns
            new_data[col] = np.full(n_h5_records, np.nan, dtype=np.float64)

    # Fill in values
    n_h5_matched = 0
    for i in range(n_h5_records):
        tic = int(tic_ids[i])
        sector = int(sectors[i])
        key = (tic, sector)

        if key in csv_lookup:
            row = csv_lookup[key]
            for col in new_columns:
                value = row[col]
                if not pd.isna(value):
                    if col in ['GaiaDR3_ID', 'cluster', 'subgroup']:
                        new_data[col][i] = str(value).encode('utf-8')
                    else:
                        new_data[col][i] = float(value)
            n_h5_matched += 1

    print(f"  Matched: {n_h5_matched}/{n_h5_records}")

    # Create backup
    backup_h5 = h5_path.with_suffix('.h5.bak')
    print(f"  Creating backup: {backup_h5}")
    import shutil
    shutil.copy(h5_path, backup_h5)

    # Add new datasets to h5 file
    print(f"  Adding new datasets to h5 file...")
    with h5py.File(h5_path, 'a') as f:
        metadata_group = f['metadata']
        for col in new_columns:
            if col in metadata_group:
                del metadata_group[col]
            metadata_group.create_dataset(col, data=new_data[col])
            print(f"    Added {col}: shape={new_data[col].shape}, dtype={new_data[col].dtype}")

    print("\nDone! Both files updated successfully.")

    # Verify
    print("\nVerification:")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    print(f"  Pickle metadata keys ({len(data['metadatas'][0].keys())}): {list(data['metadatas'][0].keys())}")

    with h5py.File(h5_path, 'r') as f:
        print(f"  H5 metadata keys ({len(f['metadata'].keys())}): {list(f['metadata'].keys())}")


if __name__ == "__main__":
    main()
