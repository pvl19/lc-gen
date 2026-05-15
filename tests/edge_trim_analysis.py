"""Quantify edge artifacts in TESS 2-min cadence light curves to assess
whether trim_edges=10 is a reasonable buffer.

Per-curve flux is z-normalised at preprocessing time, so the interior
baseline is theoretical:
    mean(f^2)    -> 1.0
    median|f|    -> 0.6745  (Gaussian)
    P(|f|>3)     -> 0.0027  (Gaussian)
Edge artifacts (scattered light, thermal settling) inflate these values.

The H5 files are chunked (256 rows x full cols, gzip), so random row reads
are extremely slow. We instead read N_CHUNKS contiguous chunks per file in
one shot, then slice rows in memory.
"""
from pathlib import Path
import h5py
import numpy as np

ROOT = Path('/Users/philvanlane/Documents/lc_ae/final_pretrain')
H5_FILES = [
    ROOT / 'timeseries_pretrain.h5',
    ROOT / 'timeseries_exop_hosts.h5',
]
ROWS_PER_CHUNK = 256        # H5 chunk row dimension
N_CHUNKS  = 8               # chunks to read per file -> 8 * 256 = 2048 samples
N_EDGE    = 40              # positions from each edge to inspect
SEED      = 0

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


def report(label, sq, mabs, emean, orate, n_edge):
    print(f'\n  --- {label} edge (col 0 = outermost) ---')
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


def analyse(path, rows_per_chunk, n_chunks, n_edge, seed):
    print(f'\n=== {path.name} ===', flush=True)
    with h5py.File(path, 'r') as h5:
        n_total = int(h5.attrs['n_samples'])
        lengths = h5['length'][:].astype(np.int64)
        n_chunks_total = (n_total + rows_per_chunk - 1) // rows_per_chunk
        rng = np.random.default_rng(seed)
        chunk_ids = np.sort(rng.choice(n_chunks_total, n_chunks, replace=False))
        print(f'  reading chunks {chunk_ids.tolist()} '
              f'(of {n_chunks_total}) -> ~{n_chunks*rows_per_chunk} samples',
              flush=True)

        flux_blocks, ferr_blocks, len_blocks = [], [], []
        for c in chunk_ids:
            r0 = int(c * rows_per_chunk)
            r1 = min(r0 + rows_per_chunk, n_total)
            print(f'    reading rows [{r0}:{r1}]', flush=True)
            flux_blocks.append(h5['flux'][r0:r1, :].astype(np.float32))
            ferr_blocks.append(h5['flux_err'][r0:r1, :].astype(np.float32))
            len_blocks.append(lengths[r0:r1])
        flux = np.concatenate(flux_blocks, axis=0)
        ferr = np.concatenate(ferr_blocks, axis=0)
        lens = np.concatenate(len_blocks, axis=0)

    n = flux.shape[0]
    print(f'  loaded {n} curves; median length={int(np.median(lens))}', flush=True)

    f_start = flux[:, :n_edge]
    e_start = ferr[:, :n_edge]
    f_end   = np.empty((n, n_edge), dtype=np.float32)
    e_end   = np.empty((n, n_edge), dtype=np.float32)
    keep    = np.zeros(n, dtype=bool)
    for i in range(n):
        L = int(lens[i])
        if L < n_edge * 2:
            f_end[i] = np.nan
            e_end[i] = np.nan
            continue
        f_end[i] = flux[i, L - n_edge:L][::-1]
        e_end[i] = ferr[i, L - n_edge:L][::-1]
        keep[i]  = True
    f_end = f_end[keep]
    e_end = e_end[keep]
    print(f'  end-edge sample: {keep.sum()} curves long enough', flush=True)

    s_sq, s_abs, s_err, s_out = per_position_stats(f_start, e_start)
    e_sq, e_abs, e_err, e_out = per_position_stats(f_end,   e_end)
    report('START', s_sq, s_abs, s_err, s_out, n_edge)
    report('END',   e_sq, e_abs, e_err, e_out, n_edge)


def main():
    for p in H5_FILES:
        if not p.exists():
            print(f'skip missing: {p}')
            continue
        analyse(p, ROWS_PER_CHUNK, N_CHUNKS, N_EDGE, SEED)


if __name__ == '__main__':
    main()
