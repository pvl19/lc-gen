"""Violin plot of host-age residuals (posterior_sample − true_age) pooled per
true-age bin.

For each true-age bin, pools posterior draws across every star in the bin and
plots the residual distribution as a violin. This shows model bias (median
offset from 0) and the spread of the posterior (width) as a function of true
age, on the same axis.

Input: a `heldout_posteriors.npz` produced by `scripts/kfold_age_inference.py`
(via `kfold_nle_age_inference_hosts.py`). If `posterior_samples` is present in
the npz it is used directly; otherwise the script samples on the fly from
`posterior` (the grid pmf) via inverse CDF, seeded for reproducibility.

Output: violin PNG next to the input npz.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_edges(s: str) -> np.ndarray:
    """Accept either a comma-separated list of edges or 'start:stop:step'."""
    s = s.strip()
    if ':' in s:
        a, b, st = (float(x) for x in s.split(':'))
        return np.arange(a, b + 1e-9, st)
    return np.array([float(x) for x in s.split(',')], dtype=float)


def draw_samples_from_grid(posterior: np.ndarray, grid: np.ndarray,
                           n_samples: int, seed: int) -> np.ndarray:
    """Inverse-CDF sampling per row. Returns (N, K) samples in `grid` units;
    NaN-filled for rows whose posterior has zero total mass."""
    N, G = posterior.shape
    rng = np.random.default_rng(seed)
    row_sum = posterior.sum(axis=1, keepdims=True)
    ok = (row_sum[:, 0] > 0) & np.isfinite(row_sum[:, 0])
    out = np.full((N, n_samples), np.nan, dtype=np.float32)
    if not ok.any():
        return out
    p_ok = posterior[ok] / row_sum[ok]
    cdf = np.cumsum(p_ok, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random((p_ok.shape[0], n_samples), dtype=np.float32)
    idx = np.empty((p_ok.shape[0], n_samples), dtype=np.int64)
    for i in range(p_ok.shape[0]):
        idx[i] = np.searchsorted(cdf[i], u[i], side='right').clip(0, G - 1)
    out[ok] = grid.astype(np.float32)[idx]
    return out


def bin_pool(residuals: np.ndarray, true_age: np.ndarray,
             edges: np.ndarray) -> tuple[list[np.ndarray], list[str], np.ndarray]:
    """For each bin defined by `edges`, pool every (star × sample) residual
    from stars whose true_age falls in the bin. Returns (data, labels, counts)
    where data[i] is the flat residual array for bin i, labels[i] is the bin
    label string, and counts[i] is the number of stars in that bin."""
    data, labels, counts = [], [], []
    K = residuals.shape[1]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (true_age >= lo) & (true_age < hi)
        # Closed-right on the final bin so the upper edge isn't dropped.
        if i == len(edges) - 2:
            in_bin |= (true_age == hi)
        n_stars = int(in_bin.sum())
        counts.append(n_stars)
        if n_stars == 0:
            data.append(np.array([]))
        else:
            flat = residuals[in_bin].reshape(-1)
            data.append(flat[np.isfinite(flat)])
        labels.append(f'{lo:g}–{hi:g}\nN={n_stars}')
    return data, labels, np.asarray(counts, dtype=np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', required=True,
                   help='Path to heldout_posteriors.npz (or its containing directory).')
    p.add_argument('--out', default=None,
                   help='Output PNG path. Default: <npz_dir>/residual_violins_by_age_bin.png')
    p.add_argument('--bin_edges', default='0,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,10,12,14',
                   help='Bin edges in Gyr. Either comma-separated ("0,0.5,1.5,...") '
                        'or numpy-style "start:stop:step". Default: 1-Gyr bins '
                        'shifted by 0.5, broadening past 8.5 Gyr where stars are sparse.')
    p.add_argument('--n_samples', type=int, default=200,
                   help='Samples per star if drawing from the grid posterior '
                        '(ignored when `posterior_samples` already exists in npz).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--age_unit', default='Gyr',
                   help='Label suffix for the axes (default: Gyr).')
    p.add_argument('--show_mean', action='store_true',
                   help='Also overplot per-bin mean residual (white circle).')
    args = p.parse_args()

    npz_path = Path(args.npz)
    if npz_path.is_dir():
        npz_path = npz_path / 'heldout_posteriors.npz'
    if not npz_path.exists():
        raise SystemExit(f'npz not found: {npz_path}')
    out_path = Path(args.out) if args.out else npz_path.parent / 'residual_violins_by_age_bin.png'

    print(f'Loading {npz_path}')
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.keys())
    print(f'  keys: {sorted(keys)}')

    if 'y_obs' not in keys:
        raise SystemExit('npz has no `y_obs` (true age) — cannot bin')
    true_age = np.asarray(d['y_obs'], dtype=np.float32)
    N = len(true_age)

    if 'posterior_samples' in keys:
        samples = np.asarray(d['posterior_samples'], dtype=np.float32)
        if samples.shape[0] != N:
            raise SystemExit(
                f'posterior_samples shape {samples.shape} does not align with y_obs ({N})')
        print(f'  using saved posterior_samples: shape={samples.shape}')
    elif 'posterior' in keys and 'grid' in keys:
        post = np.asarray(d['posterior'], dtype=np.float64)
        grid = np.asarray(d['grid'], dtype=np.float64)
        print(f'  posterior_samples not saved — sampling {args.n_samples}/star from grid '
              f'(shape={post.shape}, seed={args.seed})')
        samples = draw_samples_from_grid(post, grid, args.n_samples, args.seed)
    else:
        raise SystemExit(
            'npz lacks both `posterior_samples` and (`posterior` + `grid`) — '
            'nothing to plot. Re-run with --n_posterior_samples or include '
            'save_heldout_posteriors=True.')

    valid = np.isfinite(true_age) & np.isfinite(samples).any(axis=1)
    n_drop = int((~valid).sum())
    if n_drop:
        print(f'  dropping {n_drop} stars with NaN true_age or all-NaN samples')
    true_age = true_age[valid]
    samples = samples[valid]

    residuals = samples - true_age[:, None]    # (N, K)

    edges = parse_edges(args.bin_edges)
    if len(edges) < 2:
        raise SystemExit('--bin_edges must define at least one bin')
    print(f'  bin edges ({args.age_unit}): {list(edges)}')

    data, labels, counts = bin_pool(residuals, true_age, edges)
    n_total = int(counts.sum())
    print(f'  pooled {n_total} stars across {len(data)} bins  '
          f'(per-bin star counts: {counts.tolist()})')

    nonempty = [(i, d_, lab) for i, (d_, lab) in enumerate(zip(data, labels)) if len(d_) > 0]
    if not nonempty:
        raise SystemExit('All bins empty — check --bin_edges vs the true-age range')
    idxs, plot_data, plot_labels = zip(*nonempty)

    fig, ax = plt.subplots(figsize=(1.0 * len(plot_data) + 2.5, 5.0))
    positions = list(range(1, len(plot_data) + 1))
    parts = ax.violinplot(plot_data, positions=positions, showmedians=True,
                          showextrema=False, widths=0.85)
    for body in parts['bodies']:
        body.set_facecolor('#4c72b0')
        body.set_edgecolor('black')
        body.set_alpha(0.6)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.2)

    if args.show_mean:
        means = [float(np.mean(d_)) for d_ in plot_data]
        ax.scatter(positions, means, marker='o', s=22, facecolor='white',
                   edgecolor='black', linewidth=0.8, zorder=3, label='mean')

    ax.axhline(0.0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_labels, fontsize=9)
    ax.set_xlabel(f'True age bin ({args.age_unit})')
    ax.set_ylabel(f'Residual: posterior sample − true ({args.age_unit})')
    title = f'{npz_path.parent.name}\nposterior residuals by true-age bin '
    title += f'({n_total} stars, {samples.shape[1]} samples/star)'
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    if args.show_mean:
        ax.legend(loc='best', fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
