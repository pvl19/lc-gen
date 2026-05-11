"""Strip (or zero) the first/last hidden-state block from cached multiscale latents.

The 1536-D multiscale vector is concat of 12 blocks of size H = 128 in this order
(see compute_multiscale_features in scripts/plot_umap_latent.py):

    [ global_mean | global_std | global_max | global_min
    | seg0_mean   | seg1_mean  | seg2_mean  | seg3_mean
    | first_hidden | last_hidden                              <-- the artifact-prone block
    | diff_mean   | diff_std ]

This script reads a cached latents.npz and writes a copy with that block either:
    --mode drop  (default): block 8..10 removed, latent dim 1536 -> 1280
    --mode zero            : block 8..10 zeroed in place, latent dim stays 1536

No model re-run required — operates on the cached npz produced by
scripts/plot_umap_latent.py or scripts/kfold_age_inference.py. All other keys
(gaia_ids, tic_ids, sectors, ages, ...) are copied through unchanged.

Usage:
    python scripts/strip_first_last_hidden.py \\
        --in_npz   final_model/parallel_fixed/e60/latents.npz \\
        --out_dir  final_model/parallel_fixed/e60_no_first_last/ \\
        [--mode drop|zero]
"""
import argparse
from pathlib import Path

import numpy as np


# Block index of (first_hidden, last_hidden) inside the 12-block multiscale vector.
FIRST_LAST_BLOCK_START = 8
FIRST_LAST_BLOCK_END   = 10  # exclusive
N_BLOCKS = 12


def find_latent_key(npz: np.lib.npyio.NpzFile) -> str:
    for k in ('latent_vectors', 'latents'):
        if k in npz.files:
            return k
    raise KeyError(f'no latent_vectors / latents key in npz; got {list(npz.files)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in_npz', required=True, help='Input latents.npz')
    ap.add_argument('--out_dir', required=True,
                    help='Output directory; latents.npz will be written inside it')
    ap.add_argument('--mode', choices=['drop', 'zero'], default='drop',
                    help='drop: remove the 256-D block (default). zero: keep dim, zero the block.')
    args = ap.parse_args()

    in_path = Path(args.in_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'latents.npz'

    z = np.load(in_path, allow_pickle=True)
    latent_key = find_latent_key(z)
    X = z[latent_key]
    n, d = X.shape
    if d % N_BLOCKS != 0:
        raise ValueError(f'latent dim {d} not divisible by {N_BLOCKS}; '
                         f'cannot infer per-block hidden size H')
    H = d // N_BLOCKS
    fl_start = FIRST_LAST_BLOCK_START * H
    fl_end   = FIRST_LAST_BLOCK_END   * H
    print(f'in : {in_path}  key={latent_key}  shape={X.shape}  H={H}')
    print(f'first/last block columns: [{fl_start}, {fl_end})  size={fl_end - fl_start}')

    if args.mode == 'drop':
        keep = np.r_[0:fl_start, fl_end:d]
        X_new = X[:, keep]
    else:  # zero
        X_new = X.copy()
        X_new[:, fl_start:fl_end] = 0.0
    print(f'out: {out_path}  shape={X_new.shape}  mode={args.mode}')

    out_kwargs = {k: z[k] for k in z.files if k != latent_key}
    out_kwargs[latent_key] = X_new.astype(X.dtype)
    np.savez(out_path, **out_kwargs)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
