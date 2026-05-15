"""Is the post-downlink bump at a fixed cadence offset, or does the aggregate
peak come from curves with bumps at random positions?

For each curve with a clean downlink gap, find the per-curve position of
max f^2 in the AFTER window and histogram it. If the per-curve argmax
clusters near the aggregate peak (pos ~31), the bump is a real time-locked
transient. If it's uniformly spread, the aggregate peak is just driven by
heavy-tailed outliers happening to land at a few positions.

Uses 4 chunks (~1024 curves) of timeseries_pretrain.h5 — enough resolution
for a histogram and runs in seconds.
"""
from pathlib import Path
import h5py
import numpy as np

PATH = Path('/Users/philvanlane/Documents/lc_ae/final_pretrain/timeseries_pretrain.h5')
CHUNKS = [0, 40, 80, 120]
ROWS_PER_CHUNK = 256
N_AFTER = 60          # extend a bit past 40 to see the recovery shape
MIN_GAP_DAYS = 0.2
AMBIG_RATIO = 0.5
THRESH_RATIO = 1.5    # also report "first pos above 1.5 * curve median(f^2)"


def main():
    with h5py.File(PATH, 'r') as h5:
        n_total = int(h5.attrs['n_samples'])
        lengths = h5['length'][:].astype(np.int64)

        argmax_positions = []
        first_above = []
        bump_amplitudes = []
        per_curve_window_means = []
        for c in CHUNKS:
            r0 = c * ROWS_PER_CHUNK
            r1 = min(r0 + ROWS_PER_CHUNK, n_total)
            print(f'reading rows [{r0}:{r1}]', flush=True)
            chunk_flux = h5['flux'][r0:r1, :]
            chunk_time = h5['time'][r0:r1, :]
            for i in range(r1 - r0):
                L = int(lengths[r0 + i])
                if L < 2 * N_AFTER + 2:
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
                if g + 1 < N_AFTER or (L - (g + 1)) < N_AFTER:
                    continue
                after = chunk_flux[i, g + 1:g + 1 + N_AFTER].astype(np.float64)
                # skip the pos-0 zero marker
                f2 = after[1:] ** 2
                # per-curve median of f^2 (gives a noise scale for this curve)
                median_f2 = float(np.median(f2))
                if median_f2 <= 0:
                    continue
                # argmax position (index in the f2 array; +1 to map back to the
                # AFTER-window position)
                am = int(np.argmax(f2)) + 1
                argmax_positions.append(am)
                bump_amplitudes.append(float(f2.max()) / median_f2)
                # first position whose f^2 exceeds 1.5 * curve median f^2
                above = np.where(f2 > THRESH_RATIO * median_f2)[0]
                first_above.append(int(above[0]) + 1 if len(above) else -1)
                per_curve_window_means.append(float(np.mean(f2)))

    a = np.asarray(argmax_positions)
    b = np.asarray(bump_amplitudes)
    print(f'\nn curves: {len(a)}')
    print(f'argmax position over [1..{N_AFTER-1}]:')
    print(f'  median = {np.median(a):.1f}  mean = {np.mean(a):.1f}  '
          f'p10 = {np.percentile(a, 10):.1f}  '
          f'p25 = {np.percentile(a, 25):.1f}  '
          f'p75 = {np.percentile(a, 75):.1f}  '
          f'p90 = {np.percentile(a, 90):.1f}')
    print(f'\nbump amplitude (max f^2 / median f^2):')
    print(f'  median = {np.median(b):.2f}  '
          f'p25 = {np.percentile(b, 25):.2f}  '
          f'p75 = {np.percentile(b, 75):.2f}  '
          f'p90 = {np.percentile(b, 90):.2f}  '
          f'p99 = {np.percentile(b, 99):.2f}')

    # histogram of argmax positions, 10-cadence bins
    edges = np.arange(0, N_AFTER + 1, 5)
    hist, _ = np.histogram(a, bins=edges)
    print(f'\nargmax position histogram (5-cadence bins):')
    for lo, hi, n in zip(edges[:-1], edges[1:], hist):
        bar = '#' * int(60 * n / hist.max())
        print(f'  [{lo:>2d},{hi:>2d})  n={n:>4d}  {bar}')

    # uniform null: if argmax were uniform over [1, N_AFTER-1], expected per
    # bin would be n_total * 5 / (N_AFTER - 1)
    n_total = len(a)
    exp = n_total * 5 / (N_AFTER - 1)
    print(f'\n(uniform-null expected count per 5-cadence bin: {exp:.0f})')


if __name__ == '__main__':
    main()
