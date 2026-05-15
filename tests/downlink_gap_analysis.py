"""Quantify settling/scatter artifacts on either side of the mid-sector
downlink gap in the pretraining H5 files.

Per-curve flux is z-normalised at preprocessing time, so a clean interior
behaves like N(0,1). Per-position aggregates around the gap inflate above
the Gaussian baseline if the spacecraft takes time to settle after the
data downlink.

For each curve we:
  1. Find the largest Δt in the valid region (`:length`).
  2. Require the gap to be > MIN_GAP_DAYS and clearly the largest (the
     second-largest gap must be < AMBIG_RATIO * largest).
  3. Require >= N_EDGE valid samples on each side of the gap.
  4. Aggregate flux / flux_err per position with col 0 = closest to the
     gap on both sides ("BEFORE" reversed, "AFTER" forward).

We process one H5 chunk at a time so peak RSS stays small (~150 MB).
"""
from pathlib import Path
import time as wall
import h5py
import numpy as np

ROOT = Path('/Users/philvanlane/Documents/lc_ae/final_pretrain')
H5_FILES = [
    ROOT / 'timeseries_pretrain.h5',
    ROOT / 'timeseries_exop_hosts.h5',
]
ROWS_PER_CHUNK = 256
N_EDGE         = 40
MIN_GAP_DAYS   = 0.2     # ~5 h; real downlinks are ~1 d, normal cadence ~0.0014 d
AMBIG_RATIO    = 0.5     # skip curve if 2nd-largest gap > AMBIG_RATIO * largest
SEED           = 0       # not used (full-population pass)

B_SQ  = 1.0
B_ABS = 0.6745
B_OUT = 0.0027


def per_position_stats(flux, ferr):
    sq       = np.nanmean(flux * flux, axis=0)
    med_abs  = np.nanmedian(np.abs(flux), axis=0)
    err_mean = np.nanmean(ferr, axis=0)
    out_rate = np.nanmean(np.abs(flux) > 3, axis=0)
    return sq, med_abs, err_mean, out_rate


def first_within(arr, baseline, frac):
    excess = arr / baseline - 1.0
    hits = np.where(excess <= frac)[0]
    return int(hits[0]) if len(hits) else -1


def report(label, sq, mabs, emean, orate, n_edge, n_curves):
    print(f'\n  --- {label} gap edge (col 0 = closest to gap) | N={n_curves} ---')
    print('  pos        ' + ' '.join(f'{p:>5d}' for p in range(n_edge)))
    print('  mean(f^2): ' + ' '.join(f'{v:5.2f}' for v in sq))
    print('  median|f|: ' + ' '.join(f'{v:5.2f}' for v in mabs))
    print('  mean(err): ' + ' '.join(f'{v:5.2f}' for v in emean))
    print('  P(|f|>3) : ' + ' '.join(f'{v:5.3f}' for v in orate))
    print(f'  baselines (interior, theoretical Gaussian): mean(f^2)={B_SQ}  '
          f'median|f|={B_ABS}  P(|f|>3)={B_OUT}')
    for frac in (0.05, 0.10, 0.20, 0.50):
        i_sq  = first_within(sq,    B_SQ,  frac)
        i_abs = first_within(mabs,  B_ABS, frac)
        i_out = first_within(orate, B_OUT, frac)
        pct = int(frac * 100)
        print(f'  first pos within +{pct:>2d}% of baseline:  '
              f'mean(f^2)={i_sq:>3d}   median|f|={i_abs:>3d}   '
              f'P(|f|>3)={i_out:>3d}')


def analyse(path, rows_per_chunk, n_edge, min_gap, ambig_ratio):
    print(f'\n=== {path.name} ===', flush=True)
    t0 = wall.time()
    with h5py.File(path, 'r') as h5:
        n_total = int(h5.attrs['n_samples'])
        lengths = h5['length'][:].astype(np.int64)
        n_chunks = (n_total + rows_per_chunk - 1) // rows_per_chunk
        print(f'  {n_total} curves in {n_chunks} chunks; processing all', flush=True)

        before_blocks_f, before_blocks_e = [], []
        after_blocks_f,  after_blocks_e  = [], []
        gap_sizes = []
        gap_positions = []      # gap_idx / length
        n_skipped_short = 0
        n_skipped_nogap = 0
        n_skipped_ambig = 0

        for c in range(n_chunks):
            r0 = c * rows_per_chunk
            r1 = min(r0 + rows_per_chunk, n_total)
            chunk_flux = h5['flux'][r0:r1, :]
            chunk_ferr = h5['flux_err'][r0:r1, :]
            chunk_time = h5['time'][r0:r1, :]
            for i in range(r1 - r0):
                L = int(lengths[r0 + i])
                if L < 2 * n_edge + 2:
                    n_skipped_short += 1
                    continue
                t = chunk_time[i, :L]
                diffs = np.diff(t)
                if not np.isfinite(diffs).all():
                    n_skipped_short += 1
                    continue
                g = int(np.argmax(diffs))     # gap between t[g] and t[g+1]
                gap_size = float(diffs[g])
                if gap_size < min_gap:
                    n_skipped_nogap += 1
                    continue
                # ambiguity check: 2nd-largest must be clearly smaller
                d2 = diffs.copy()
                d2[g] = -np.inf
                if float(d2.max()) > ambig_ratio * gap_size:
                    n_skipped_ambig += 1
                    continue
                # need n_edge samples on each side
                if g + 1 < n_edge or (L - (g + 1)) < n_edge:
                    n_skipped_short += 1
                    continue
                before_f = chunk_flux[i, g + 1 - n_edge:g + 1][::-1]
                before_e = chunk_ferr[i, g + 1 - n_edge:g + 1][::-1]
                after_f  = chunk_flux[i, g + 1:g + 1 + n_edge]
                after_e  = chunk_ferr[i, g + 1:g + 1 + n_edge]
                before_blocks_f.append(before_f)
                before_blocks_e.append(before_e)
                after_blocks_f.append(after_f)
                after_blocks_e.append(after_e)
                gap_sizes.append(gap_size)
                gap_positions.append((g + 1) / L)
            if (c + 1) % 10 == 0 or c + 1 == n_chunks:
                elapsed = wall.time() - t0
                print(f'  processed {c + 1}/{n_chunks} chunks  '
                      f'kept={len(before_blocks_f)}  '
                      f'skip_short={n_skipped_short}  '
                      f'skip_nogap={n_skipped_nogap}  '
                      f'skip_ambig={n_skipped_ambig}  '
                      f'({elapsed:.1f}s)',
                      flush=True)

    n_kept = len(before_blocks_f)
    print(f'\n  kept {n_kept}/{n_total} curves with a clean downlink-class gap',
          flush=True)
    if n_kept == 0:
        return
    gap_sizes = np.asarray(gap_sizes)
    gap_positions = np.asarray(gap_positions)
    print(f'  gap size (days):     median={np.median(gap_sizes):.3f}  '
          f'p10={np.percentile(gap_sizes, 10):.3f}  '
          f'p90={np.percentile(gap_sizes, 90):.3f}  '
          f'max={gap_sizes.max():.3f}')
    print(f'  gap position (frac): median={np.median(gap_positions):.3f}  '
          f'p10={np.percentile(gap_positions, 10):.3f}  '
          f'p90={np.percentile(gap_positions, 90):.3f}')

    bf = np.stack(before_blocks_f, axis=0)
    be = np.stack(before_blocks_e, axis=0)
    af = np.stack(after_blocks_f,  axis=0)
    ae = np.stack(after_blocks_e,  axis=0)

    s_sq, s_abs, s_err, s_out = per_position_stats(bf, be)
    e_sq, e_abs, e_err, e_out = per_position_stats(af, ae)
    report('BEFORE (trailing edge of pre-gap segment)',
           s_sq, s_abs, s_err, s_out, n_edge, n_kept)
    report('AFTER  (leading edge of post-gap segment)',
           e_sq, e_abs, e_err, e_out, n_edge, n_kept)

    print(f'\n  total wall: {wall.time() - t0:.1f}s', flush=True)


def main():
    for p in H5_FILES:
        if not p.exists():
            print(f'skip missing: {p}')
            continue
        analyse(p, ROWS_PER_CHUNK, N_EDGE, MIN_GAP_DAYS, AMBIG_RATIO)


if __name__ == '__main__':
    main()
