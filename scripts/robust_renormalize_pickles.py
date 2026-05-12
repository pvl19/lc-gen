"""Re-normalize cached pickles with robust statistics (median, (p84-p16)/2).

The original download notebook stored flux as a z-score using mean / std, both
of which are pulled around by a single extreme outlier. This script recovers
the raw flux per light curve via the saved mean_flux/std_flux metadata,
recomputes robust scaling, and writes new pickles. Original pickles are not
modified. Raw flux_err NaN positions are preserved (NaN/scalar = NaN).

Each output record stores:
    flux            (median-centred, iqr-half-scaled)
    flux_err        (raw / iqr_half; NaN positions preserved)
    metadata.median_flux     (new)
    metadata.iqr_half_flux   (new)
    metadata.mean_flux       (unchanged — describes the OLD scaling)
    metadata.std_flux        (unchanged)
All other keys pass through.

Usage:
    python scripts/robust_renormalize_pickles.py \\
        --in_pickles final_pretrain/lc_data_p1.pickle ... \\
        --out_dir   final_pretrain_robust/
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def renormalize_record(rec: dict) -> dict:
    md = rec['metadata']
    mean_old = float(md['mean_flux'])
    std_old  = float(md['std_flux'])

    # Lossless inversion of the original z-score normalization.
    flux_raw     = np.asarray(rec['flux'],     dtype=np.float64) * std_old + mean_old
    flux_err_raw = np.asarray(rec['flux_err'], dtype=np.float64) * std_old

    # Robust scaling computed on finite values only.
    finite = np.isfinite(flux_raw)
    if not finite.any():
        raise ValueError('no finite raw flux samples after inversion')
    f_finite = flux_raw[finite]
    median_flux = float(np.median(f_finite))
    p84, p16 = np.percentile(f_finite, [84, 16])
    iqr_half = float((p84 - p16) / 2.0)
    if not np.isfinite(iqr_half) or iqr_half <= 0:
        raise ValueError(f'degenerate iqr_half={iqr_half}')

    flux_new     = (flux_raw - median_flux) / iqr_half
    flux_err_new = flux_err_raw / iqr_half          # NaN positions propagate

    out = dict(rec)
    out['flux']     = flux_new.astype(np.float32)
    out['flux_err'] = flux_err_new.astype(np.float32)
    out_md = dict(md)
    out_md['median_flux']   = median_flux
    out_md['iqr_half_flux'] = iqr_half
    out['metadata'] = out_md
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in_pickles', required=True, nargs='+',
                    help='One or more input pickles (e.g. lc_data_p1.pickle ...).')
    ap.add_argument('--out_dir', required=True,
                    help='Output directory; one file per input, same basename.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_str in args.in_pickles:
        in_path = Path(in_str)
        print(f'\n== {in_path} ==')
        with open(in_path, 'rb') as f:
            data = pickle.load(f)
        print(f'  loaded {len(data)} records')

        new_data = {}
        skipped = 0
        ratios = []   # iqr_half / std_old, for sanity reporting
        for key, rec in data.items():
            try:
                new_rec = renormalize_record(rec)
            except Exception as e:
                print(f'  skip {key}: {e}')
                skipped += 1
                continue
            new_data[key] = new_rec
            ratios.append(new_rec['metadata']['iqr_half_flux']
                          / float(rec['metadata']['std_flux']))

        ratios = np.asarray(ratios)
        if ratios.size:
            print(f'  iqr_half / std_old  median={np.median(ratios):.3f}  '
                  f'min={ratios.min():.3f}  max={ratios.max():.3f}  '
                  f'(ratios well below 1.0 indicate std was inflated by outliers)')
        print(f'  skipped {skipped} record(s)')

        out_path = out_dir / in_path.name
        with open(out_path, 'wb') as f:
            pickle.dump(new_data, f)
        print(f'  wrote {out_path}  ({len(new_data)} records)')


if __name__ == '__main__':
    main()
