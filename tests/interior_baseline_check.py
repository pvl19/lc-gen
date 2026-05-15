"""How elevated are the gap-edge windows compared to the SAME curves'
clean interior?

For each curve with a clean downlink gap, we take the centre 50% of each
sub-segment (i.e. the middle of pre-gap and middle of post-gap) — well
away from both the sector edge and the downlink — as the interior
baseline. Then we compare the gap-edge mean(f^2) (positions 1..40 on
each side, skipping the zero marker) against that per-curve interior.

If interior ~= gap-edge, the only real artifact is the pos-0 zero marker.
If gap-edge >> interior, there is real settling/scatter elevation.
"""
from pathlib import Path
import h5py
import numpy as np

PATH = Path('/Users/philvanlane/Documents/lc_ae/final_pretrain/timeseries_pretrain.h5')
CHUNKS = [0, 40, 80, 120]
ROWS_PER_CHUNK = 256
N_EDGE = 40
MIN_GAP_DAYS = 0.2
AMBIG_RATIO = 0.5

EDGE_BUFFER = 50    # also stay >= EDGE_BUFFER samples away from sector ends


def main():
    with h5py.File(PATH, 'r') as h5:
        n_total = int(h5.attrs['n_samples'])
        lengths = h5['length'][:].astype(np.int64)

        # per-curve arrays
        before_edge_meanf2  = []
        before_int_meanf2   = []
        after_edge_meanf2   = []
        after_int_meanf2    = []
        whole_curve_meanf2  = []

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
                    continue
                d2 = diffs.copy()
                d2[g] = -np.inf
                if float(d2.max()) > AMBIG_RATIO * float(diffs[g]):
                    continue
                # need room for edge windows
                if g + 1 < N_EDGE + EDGE_BUFFER + 100 or \
                   (L - (g + 1)) < N_EDGE + EDGE_BUFFER + 100:
                    continue
                f = chunk_flux[i, :L].astype(np.float64)

                # gap-edge windows (skip pos 0 zero marker)
                bef_edge = f[g + 1 - N_EDGE:g]   # last N_EDGE-1 pre-gap samples
                aft_edge = f[g + 2:g + 1 + N_EDGE]  # first N_EDGE-1 post-gap

                # interior of pre-gap segment: centre 50%, but stay >=
                # EDGE_BUFFER from sector start and from gap
                pre_lo = max(EDGE_BUFFER, (g + 1) // 4)
                pre_hi = min(g + 1 - EDGE_BUFFER, 3 * (g + 1) // 4)
                if pre_hi - pre_lo < 100:
                    continue
                bef_int = f[pre_lo:pre_hi]

                # interior of post-gap segment
                post_len = L - (g + 1)
                post_lo = (g + 1) + max(EDGE_BUFFER, post_len // 4)
                post_hi = (g + 1) + min(post_len - EDGE_BUFFER, 3 * post_len // 4)
                if post_hi - post_lo < 100:
                    continue
                aft_int = f[post_lo:post_hi]

                before_edge_meanf2.append(float(np.mean(bef_edge**2)))
                before_int_meanf2.append(float(np.mean(bef_int**2)))
                after_edge_meanf2.append(float(np.mean(aft_edge**2)))
                after_int_meanf2.append(float(np.mean(aft_int**2)))
                whole_curve_meanf2.append(float(np.mean(f**2)))

    bef_e = np.asarray(before_edge_meanf2)
    bef_i = np.asarray(before_int_meanf2)
    aft_e = np.asarray(after_edge_meanf2)
    aft_i = np.asarray(after_int_meanf2)
    whole = np.asarray(whole_curve_meanf2)
    n = len(bef_e)
    print(f'\nn curves: {n}\n')

    def summary(label, arr):
        print(f'  {label:30s} median={np.median(arr):.3f}  '
              f'mean={np.mean(arr):.3f}  '
              f'p25={np.percentile(arr,25):.3f}  '
              f'p75={np.percentile(arr,75):.3f}')

    summary('whole-curve mean(f^2)',          whole)
    summary('pre-gap interior mean(f^2)',     bef_i)
    summary('pre-gap edge mean(f^2)  (1..39)',bef_e)
    summary('post-gap interior mean(f^2)',    aft_i)
    summary('post-gap edge mean(f^2) (1..39)',aft_e)

    print('\n  per-curve ratios (gap-edge / own interior):')
    summary('  BEFORE edge / BEFORE interior', bef_e / bef_i)
    summary('  AFTER  edge / AFTER  interior', aft_e / aft_i)

    # what fraction of curves have edge > k * interior for various k
    print('\n  cumulative fraction of curves with edge > k * interior:')
    for k in (1.0, 1.1, 1.25, 1.5, 2.0):
        bf = float(np.mean(bef_e > k * bef_i))
        af = float(np.mean(aft_e > k * aft_i))
        print(f'   k={k:>4.2f}   BEFORE: {bf:.3f}   AFTER: {af:.3f}')


if __name__ == '__main__':
    main()
