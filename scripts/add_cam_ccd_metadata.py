#!/usr/bin/env python
"""
Add Tmag, camera, and ccd from data/cf_cam_ccds.csv into the metadata group
of data/timeseries_x.h5, matched by (TIC_ID, sector).
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = Path("data/cf_cam_ccds.csv")
H5_PATH  = Path("data/timeseries_x.h5")


def main():
    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows, columns: {df.columns.tolist()}")

    # Build (TIC_ID, sector) → row lookup
    lookup = {(int(r.TIC_ID), int(r.sector)): r for _, r in df.iterrows()}
    print(f"  Lookup size: {len(lookup)}")

    # ── Read H5 keys ──────────────────────────────────────────────────────────
    print(f"\nReading H5 ids from {H5_PATH} ...")
    with h5py.File(H5_PATH, 'r', locking=False) as f:
        tic_ids = f['metadata']['tic'][:]
        sectors  = f['metadata']['sector'][:].astype(int)
    N = len(tic_ids)
    print(f"  {N} records in H5")

    # ── Build arrays ──────────────────────────────────────────────────────────
    tmag   = np.full(N, np.nan, dtype=np.float32)
    camera = np.full(N, -1,    dtype=np.int32)
    ccd    = np.full(N, -1,    dtype=np.int32)

    n_matched = 0
    for i in range(N):
        key = (int(tic_ids[i]), int(sectors[i]))
        row = lookup.get(key)
        if row is not None:
            tmag[i]   = float(row.Tmag)   if not pd.isna(row.Tmag)   else np.nan
            camera[i] = int(row.camera)   if not pd.isna(row.camera) else -1
            ccd[i]    = int(row.ccd)      if not pd.isna(row.ccd)    else -1
            n_matched += 1

    print(f"  Matched {n_matched}/{N} records")
    n_missing = (camera == -1).sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} records without a CSV match (camera/ccd = -1)")

    # ── Write to H5 ───────────────────────────────────────────────────────────
    print(f"\nWriting to {H5_PATH} ...")
    with h5py.File(H5_PATH, 'a', locking=False) as f:
        meta = f['metadata']

        # Overwrite Tmag if it exists, else create
        if 'Tmag' in meta:
            meta['Tmag'][:] = tmag
            print("  Overwrote 'Tmag'")
        else:
            meta.create_dataset('Tmag', data=tmag)
            print("  Created 'Tmag'")

        # camera
        if 'camera' in meta:
            meta['camera'][:] = camera
            print("  Overwrote 'camera'")
        else:
            meta.create_dataset('camera', data=camera)
            print("  Created 'camera'")

        # ccd
        if 'ccd' in meta:
            meta['ccd'][:] = ccd
            print("  Overwrote 'ccd'")
        else:
            meta.create_dataset('ccd', data=ccd)
            print("  Created 'ccd'")

    # ── Verify ────────────────────────────────────────────────────────────────
    print("\nVerification:")
    with h5py.File(H5_PATH, 'r', locking=False) as f:
        meta = f['metadata']
        print(f"  metadata keys: {sorted(meta.keys())}")
        print(f"  Tmag  : min={meta['Tmag'][:].min():.2f}  max={meta['Tmag'][:].max():.2f}  "
              f"nan_frac={np.isnan(meta['Tmag'][:]).mean():.3f}")
        print(f"  camera: unique={np.unique(meta['camera'][:])}")
        print(f"  ccd   : unique={np.unique(meta['ccd'][:])}")

    print("\nDone.")


if __name__ == "__main__":
    main()
