"""Generate a noise-baseline latents npz.

Takes an existing latents npz (e.g. final_model/sendit/e100/latents_hosts_merged.npz)
and writes a parallel file where `latent_vectors` is replaced by per-feature
mean/std-matched Gaussian noise. All other fields (gaia_ids, sectors, ages,
bprp0, tic_ids, subsector, ...) are copied verbatim so downstream code joins
identically — only the feature content is randomized.

Purpose: a control baseline that uses the *same architecture* as the real
latent run (MLP encoder → flow), with inputs that have matched first-order
statistics but no age information. If the real model beats this baseline,
the latents are doing more than provide noise of the right scale.

Defaults are deterministic given --seed.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in_npz', required=True,
                   help='Source latents npz (must contain `latent_vectors`).')
    p.add_argument('--out_npz', required=True,
                   help='Destination noise-baseline npz.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--match', choices=['per_feature', 'unit'], default='per_feature',
                   help="'per_feature' (default): noise ~ N(mu_d, sigma_d) per "
                        "feature d, matched to the empirical mean/std of the "
                        "input latents. 'unit': noise ~ N(0, 1) regardless of "
                        "input scale (probes whether the MLP's first layer "
                        "depends on input statistics).")
    p.add_argument('--force', action='store_true',
                   help='Overwrite --out_npz if it exists.')
    args = p.parse_args()

    in_path = Path(args.in_npz)
    out_path = Path(args.out_npz)
    if not in_path.exists():
        raise SystemExit(f'input not found: {in_path}')
    if out_path.exists() and not args.force:
        raise SystemExit(f'{out_path} exists — pass --force to overwrite')

    print(f'Loading {in_path}')
    d = dict(np.load(in_path, allow_pickle=True))
    if 'latent_vectors' not in d:
        raise SystemExit("input npz has no `latent_vectors` field")
    lv = np.asarray(d['latent_vectors'])
    N, D = lv.shape
    print(f'  latent shape: {lv.shape}, dtype={lv.dtype}')
    print(f'  N stars / rows: {N}, feature dim: {D}')

    rng = np.random.default_rng(args.seed)
    if args.match == 'per_feature':
        # Per-feature empirical mean/std (matches both NaN-free and finite
        # entries; latents are dense float32, no NaNs expected).
        mu = lv.mean(axis=0)          # (D,)
        sd = lv.std(axis=0)           # (D,)
        sd_safe = np.where(sd > 0, sd, 1.0)
        print(f'  per-feature stats: mu range [{mu.min():.3g}, {mu.max():.3g}], '
              f'sigma range [{sd.min():.3g}, {sd.max():.3g}]')
        noise = rng.standard_normal((N, D)).astype(lv.dtype)
        noise = noise * sd_safe.astype(lv.dtype) + mu.astype(lv.dtype)
    else:  # 'unit'
        print('  matching: unit Gaussian (N(0,1)) — ignores input scale')
        noise = rng.standard_normal((N, D)).astype(lv.dtype)

    d['latent_vectors'] = noise

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **d)
    print(f'\nWrote {out_path}')
    print(f'  fields preserved: {sorted(d.keys())}')
    print(f'  seed: {args.seed}, match: {args.match}')


if __name__ == '__main__':
    main()
