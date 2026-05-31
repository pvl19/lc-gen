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


def load_residuals(npz_path: Path, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Load a heldout_posteriors.npz, return (residuals, true_age, K) where
    residuals has shape (N, K). Prefers saved posterior_samples; falls back to
    inverse-CDF sampling from the grid posterior.
    """
    print(f'Loading {npz_path}')
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.keys())
    print(f'  keys: {sorted(keys)}')
    if 'y_obs' not in keys:
        raise SystemExit(f'{npz_path}: no `y_obs` — cannot bin')
    true_age = np.asarray(d['y_obs'], dtype=np.float32)

    if 'posterior_samples' in keys:
        samples = np.asarray(d['posterior_samples'], dtype=np.float32)
        if samples.shape[0] != len(true_age):
            raise SystemExit(f'{npz_path}: posterior_samples shape {samples.shape} '
                             f'does not align with y_obs ({len(true_age)})')
        print(f'  using saved posterior_samples: shape={samples.shape}')
    elif 'posterior' in keys and 'grid' in keys:
        post = np.asarray(d['posterior'], dtype=np.float64)
        grid = np.asarray(d['grid'], dtype=np.float64)
        print(f'  posterior_samples not saved — sampling {n_samples}/star from grid '
              f'(shape={post.shape}, seed={seed})')
        samples = draw_samples_from_grid(post, grid, n_samples, seed)
    else:
        raise SystemExit(
            f'{npz_path}: lacks both `posterior_samples` and (`posterior` + `grid`) — '
            'nothing to plot.')

    valid = np.isfinite(true_age) & np.isfinite(samples).any(axis=1)
    n_drop = int((~valid).sum())
    if n_drop:
        print(f'  dropping {n_drop} stars with NaN true_age or all-NaN samples')
    true_age = true_age[valid]
    samples = samples[valid]
    return samples - true_age[:, None], true_age, samples.shape[1]


def draw_violin_group(ax, plot_data, positions, face_color, label,
                      width, show_mean):
    parts = ax.violinplot(plot_data, positions=positions, showmedians=True,
                          showextrema=False, widths=width)
    for body in parts['bodies']:
        body.set_facecolor(face_color)
        body.set_edgecolor('black')
        body.set_alpha(0.65)
        body.set_linewidth(0.6)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.0)
    handle = plt.matplotlib.patches.Patch(
        facecolor=face_color, edgecolor='black', alpha=0.65, label=label)
    if show_mean:
        means = [float(np.mean(d_)) if len(d_) else np.nan for d_ in plot_data]
        ax.scatter(positions, means, marker='o', s=20, facecolor='white',
                   edgecolor='black', linewidth=0.8, zorder=3)
    return handle


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', required=True,
                   help='Path to heldout_posteriors.npz (or its containing directory).')
    p.add_argument('--noise_npz', default=None,
                   help='Optional second heldout_posteriors.npz from the noise-baseline '
                        'run. When given, its residuals are plotted as a second violin '
                        'per bin (side-by-side) so the two are directly comparable.')
    p.add_argument('--out', default=None,
                   help='Output PNG path. Default: <npz_dir>/residual_violins_by_age_bin'
                        '[_vs_noise].png')
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
    p.add_argument('--latent_label', default='latent',
                   help='Legend label for the primary --npz violins (default: "latent").')
    p.add_argument('--noise_label', default='noise baseline',
                   help='Legend label for the --noise_npz violins (default: "noise baseline").')
    args = p.parse_args()

    def resolve(path_str):
        p_ = Path(path_str)
        if p_.is_dir():
            p_ = p_ / 'heldout_posteriors.npz'
        if not p_.exists():
            raise SystemExit(f'npz not found: {p_}')
        return p_

    npz_path = resolve(args.npz)
    noise_path = resolve(args.noise_npz) if args.noise_npz else None

    edges = parse_edges(args.bin_edges)
    if len(edges) < 2:
        raise SystemExit('--bin_edges must define at least one bin')
    print(f'\nbin edges ({args.age_unit}): {list(edges)}\n')

    # Primary (latent) series.
    residuals_a, true_age_a, K_a = load_residuals(npz_path, args.n_samples, args.seed)
    data_a, labels_a, counts_a = bin_pool(residuals_a, true_age_a, edges)
    print(f'  pooled {int(counts_a.sum())} stars across {len(data_a)} bins  '
          f'(per-bin: {counts_a.tolist()})\n')

    # Optional noise series (must use the SAME bin edges so positions align).
    data_b = labels_b = counts_b = K_b = None
    if noise_path is not None:
        residuals_b, true_age_b, K_b = load_residuals(noise_path, args.n_samples, args.seed)
        data_b, labels_b, counts_b = bin_pool(residuals_b, true_age_b, edges)
        print(f'  pooled {int(counts_b.sum())} stars across {len(data_b)} bins  '
              f'(per-bin: {counts_b.tolist()})\n')

    # Default output filename reflects single vs. comparison mode.
    if args.out:
        out_path = Path(args.out)
    else:
        stem = 'residual_violins_by_age_bin' + ('_vs_noise' if noise_path else '')
        out_path = npz_path.parent / f'{stem}.png'

    # Drop bins that are empty in BOTH series (keeps positions consistent).
    keep = [(len(d_) > 0) or (data_b is not None and len(data_b[i]) > 0)
            for i, d_ in enumerate(data_a)]
    if not any(keep):
        raise SystemExit('All bins empty — check --bin_edges vs the true-age range.')
    idxs = [i for i, k in enumerate(keep) if k]
    plot_labels = [labels_a[i] for i in idxs]
    plot_data_a = [data_a[i] for i in idxs]
    plot_data_b = [data_b[i] for i in idxs] if data_b is not None else None

    fig, ax = plt.subplots(figsize=(1.05 * len(idxs) + 2.8, 5.2))
    positions = np.arange(1, len(idxs) + 1)

    if plot_data_b is None:
        h_a = draw_violin_group(ax, plot_data_a, positions, '#4c72b0',
                                args.latent_label, width=0.85,
                                show_mean=args.show_mean)
        handles = [h_a]
    else:
        # Side-by-side violins per bin: offset by ±0.22; widths shrunk to 0.42.
        off = 0.22
        h_a = draw_violin_group(ax, plot_data_a, positions - off, '#4c72b0',
                                args.latent_label, width=0.42,
                                show_mean=args.show_mean)
        h_b = draw_violin_group(ax, plot_data_b, positions + off, '#bdbdbd',
                                args.noise_label, width=0.42,
                                show_mean=args.show_mean)
        handles = [h_a, h_b]

    ax.axhline(0.0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_labels, fontsize=9)
    ax.set_xlabel(f'True age bin ({args.age_unit})')
    ax.set_ylabel(f'Residual: posterior sample − true ({args.age_unit})')

    title = f'{npz_path.parent.name}'
    if noise_path is not None:
        title += f'  vs  {noise_path.parent.name}'
    title += (f'\nposterior residuals by true-age bin '
              f'({int(counts_a.sum())} stars, {K_a} samples/star)')
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(handles=handles, loc='best', fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
