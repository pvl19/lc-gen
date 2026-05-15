"""Verify the sector-edge findings using the same per-curve comparison
that flipped the downlink result.

For each curve we compute:
  - first-N mean(f^2)  (start sector edge)
  - last-N mean(f^2)   (end sector edge)
  - interior mean(f^2) (centre 50% of each sub-segment around the gap)

and report per-curve ratios edge/interior.

If the median ratio stays ~1, the sector-edge "elevation" was also an
aggregate-vs-Gaussian artifact and the trim_edges recommendation needs
to be retracted. If the median ratio is well above 1, the sector-edge
finding stands.
"""
from pathlib import Path
import h5py
import numpy as np

PATH = Path('/Users/philvanlane/Documents/lc_ae/final_pretrain/timeseries_pretrain.h5')
CHUNKS = [0, 40, 80, 120]
ROWS_PER_CHUNK = 256
N_EDGE = 40
MIN_GAP_DAYS = 0.2
EDGE_BUFFER = 50


def main():
    with h5py.File(PATH, 'r') as h5:
        n_total = int(h5.attrs['n_samples'])
        lengths = h5['length'][:].astype(np.int64)

        start_edge_meanf2 = []
        end_edge_meanf2   = []
        interior_meanf2   = []
        # also slice the start edge by position bucket so we can see the
        # decay shape per-curve (vs the aggregate)
        first_5_meanf2    = []   # first 5 samples
        first_10_meanf2   = []   # first 10 samples
        first_20_meanf2   = []   # first 20 samples
        last_10_meanf2    = []
        last_20_meanf2    = []

        for c in CHUNKS:
            r0 = c * ROWS_PER_CHUNK
            r1 = min(r0 + ROWS_PER_CHUNK, n_total)
            print(f'reading rows [{r0}:{r1}]', flush=True)
            chunk_flux = h5['flux'][r0:r1, :]
            chunk_time = h5['time'][r0:r1, :]
            for i in range(r1 - r0):
                L = int(lengths[r0 + i])
                if L < 2 * (N_EDGE + EDGE_BUFFER + 200):
                    continue
                t = chunk_time[i, :L]
                diffs = np.diff(t)
                if not np.isfinite(diffs).all():
                    continue
                g = int(np.argmax(diffs))
                if float(diffs[g]) < MIN_GAP_DAYS:
                    g = -1   # no big gap; treat whole curve as one segment

                f = chunk_flux[i, :L].astype(np.float64)

                # interior: centre 50% of each sub-segment, away from edges
                interior_chunks = []
                if g > 0:
                    pre_lo  = max(EDGE_BUFFER, (g + 1) // 4)
                    pre_hi  = min(g + 1 - EDGE_BUFFER, 3 * (g + 1) // 4)
                    if pre_hi - pre_lo >= 100:
                        interior_chunks.append(f[pre_lo:pre_hi])
                    post_len = L - (g + 1)
                    post_lo = (g + 1) + max(EDGE_BUFFER, post_len // 4)
                    post_hi = (g + 1) + min(post_len - EDGE_BUFFER,
                                             3 * post_len // 4)
                    if post_hi - post_lo >= 100:
                        interior_chunks.append(f[post_lo:post_hi])
                else:
                    lo, hi = max(EDGE_BUFFER, L // 4), min(L - EDGE_BUFFER,
                                                            3 * L // 4)
                    if hi - lo >= 100:
                        interior_chunks.append(f[lo:hi])
                if not interior_chunks:
                    continue
                interior = np.concatenate(interior_chunks)

                start_edge_meanf2.append(float(np.mean(f[:N_EDGE]**2)))
                end_edge_meanf2.append  (float(np.mean(f[L-N_EDGE:L]**2)))
                interior_meanf2.append  (float(np.mean(interior**2)))
                first_5_meanf2.append   (float(np.mean(f[:5]**2)))
                first_10_meanf2.append  (float(np.mean(f[:10]**2)))
                first_20_meanf2.append  (float(np.mean(f[:20]**2)))
                last_10_meanf2.append   (float(np.mean(f[L-10:L]**2)))
                last_20_meanf2.append   (float(np.mean(f[L-20:L]**2)))

    s  = np.asarray(start_edge_meanf2)
    e  = np.asarray(end_edge_meanf2)
    inter = np.asarray(interior_meanf2)
    n  = len(s)
    print(f'\nn curves: {n}\n')

    def summary(label, arr):
        print(f'  {label:38s} median={np.median(arr):.3f}  '
              f'mean={np.mean(arr):.3f}  '
              f'p25={np.percentile(arr,25):.3f}  '
              f'p75={np.percentile(arr,75):.3f}')

    summary('interior mean(f^2)',                   inter)
    summary('start edge (first 40) mean(f^2)',      s)
    summary('end   edge (last 40)  mean(f^2)',      e)

    print('\n  per-curve edge / interior ratios:')
    summary('  first 5  / interior',  np.asarray(first_5_meanf2)  / inter)
    summary('  first 10 / interior',  np.asarray(first_10_meanf2) / inter)
    summary('  first 20 / interior',  np.asarray(first_20_meanf2) / inter)
    summary('  first 40 / interior',  s / inter)
    summary('  last  10 / interior',  np.asarray(last_10_meanf2)  / inter)
    summary('  last  20 / interior',  np.asarray(last_20_meanf2)  / inter)
    summary('  last  40 / interior',  e / inter)

    print('\n  cumulative fraction of curves with start-edge > k * interior:')
    for k in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
        f5  = float(np.mean(np.asarray(first_5_meanf2)  > k * inter))
        f10 = float(np.mean(np.asarray(first_10_meanf2) > k * inter))
        f20 = float(np.mean(np.asarray(first_20_meanf2) > k * inter))
        print(f'   k={k:>4.2f}   first5: {f5:.3f}   first10: {f10:.3f}   first20: {f20:.3f}')

    print('\n  cumulative fraction of curves with end-edge > k * interior:')
    for k in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
        l10 = float(np.mean(np.asarray(last_10_meanf2) > k * inter))
        l20 = float(np.mean(np.asarray(last_20_meanf2) > k * inter))
        print(f'   k={k:>4.2f}   last10: {l10:.3f}   last20: {l20:.3f}')


if __name__ == '__main__':
    main()
