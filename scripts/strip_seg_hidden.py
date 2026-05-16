"""Strip (or zero) the seg0..seg3 blocks from cached multiscale latents.

The 1536-D multiscale vector is concat of 12 blocks of size H = 128 in this order
(see compute_multiscale_features in scripts/plot_umap_latent.py):

    [ global_mean | global_std | global_max | global_min
    | seg0_mean   | seg1_mean  | seg2_mean  | seg3_mean   <-- equal-time bin pool
    | first_hidden | last_hidden
    | diff_mean   | diff_std ]

Even with the time-aware bin definition (Route A), an entire near-empty time
quartile collapses that block's L2 norm to ~0 and creates a discrete UMAP
fingerprint — see output/umap/umap_analysis.ipynb sector 77 / s42-54-91 blobs.
This script drops the four seg blocks so we can test whether glob_* + diff_* +
first/last alone keep age accuracy without producing the gap-driven islands.

    --mode drop  (default): seg0..seg3 removed, latent dim 1536 -> 1024
    --mode zero            : seg0..seg3 zeroed in place, latent dim stays 1536

No model re-run required — operates on the cached npz produced by
scripts/plot_umap_latent.py or scripts/kfold_age_inference.py. All other keys
(gaia_ids, tic_ids, sectors, ages, ...) are copied through unchanged.

Usage:
    python scripts/strip_seg_hidden.py \\
        --in_npz   final_model/parallel_fixed/e110_timeaware/latents_pretrain.npz \\
                   final_model/parallel_fixed/e110_timeaware/latents_hosts.npz \\
        --out_dir  final_model/parallel_fixed/e110_no_seg/ \\
        [--mode drop|zero]
"""
import argparse
from pathlib import Path

import numpy as np


# Block indices of (seg0, seg1, seg2, seg3) inside the 12-block multiscale vector.
SEG_BLOCK_START = 4
SEG_BLOCK_END   = 8  # exclusive
N_BLOCKS = 12


def find_latent_key(npz: np.lib.npyio.NpzFile) -> str:
    for k in ('latent_vectors', 'latents'):
        if k in npz.files:
            return k
    raise KeyError(f'no latent_vectors / latents key in npz; got {list(npz.files)}')


def process_one(in_path: Path, out_path: Path, mode: str) -> None:
    z = np.load(in_path, allow_pickle=True)
    latent_key = find_latent_key(z)
    X = z[latent_key]
    n, d = X.shape
    if d % N_BLOCKS != 0:
        raise ValueError(f'{in_path}: latent dim {d} not divisible by {N_BLOCKS}; '
                         f'cannot infer per-block hidden size H')
    H = d // N_BLOCKS
    seg_start = SEG_BLOCK_START * H
    seg_end   = SEG_BLOCK_END   * H
    print(f'in : {in_path}  key={latent_key}  shape={X.shape}  H={H}')
    print(f'seg block columns: [{seg_start}, {seg_end})  size={seg_end - seg_start}')

    if mode == 'drop':
        keep = np.r_[0:seg_start, seg_end:d]
        X_new = X[:, keep]
    else:  # zero
        X_new = X.copy()
        X_new[:, seg_start:seg_end] = 0.0
    print(f'out: {out_path}  shape={X_new.shape}  mode={mode}')

    out_kwargs = {k: z[k] for k in z.files if k != latent_key}
    out_kwargs[latent_key] = X_new.astype(X.dtype)
    np.savez(out_path, **out_kwargs)
    print(f'wrote {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in_npz', required=True, nargs='+',
                    help='One or more input latents npz files (e.g. latents_pretrain.npz '
                         'latents_hosts.npz). Each input basename is preserved in --out_dir.')
    ap.add_argument('--out_dir', required=True,
                    help='Output directory; one output file per input, same basename.')
    ap.add_argument('--mode', choices=['drop', 'zero'], default='drop',
                    help='drop: remove the 512-D block (default). zero: keep dim, zero the block.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_str in args.in_npz:
        in_path = Path(in_str)
        out_path = out_dir / in_path.name
        process_one(in_path, out_path, args.mode)


if __name__ == '__main__':
    main()
