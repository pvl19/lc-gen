"""Plot per-timestep saliency for a single star.

Consumes the .npz produced by compute_saliency.py and renders three panels
sharing the time axis:

  Top:    flux trace, points colored by signed IG attribution (||z||^2 target
          by default; --target pc1 switches to the PC1 target if present).
  Middle: flux_err trace, points colored by signed IG attribution.
  Bottom: minGRU gate-weight w_t over time (line + fill, no per-point color).

Output: PNG into the same directory as the input npz.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm


def _colored_segments(ax, t, y, s, cmap, norm, lw=0.8, alpha=1.0):
    """Draw y vs t as connected line segments colored by per-point value s."""
    pts = np.column_stack([t, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)        # (L-1, 2, 2)
    # Per-segment color: average the two endpoint attributions.
    s_seg = 0.5 * (s[:-1] + s[1:])
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw, alpha=alpha)
    lc.set_array(s_seg)
    ax.add_collection(lc)
    return lc


def _percentile_norm(s, q=99.0):
    """Symmetric two-slope diverging norm sized to the q-th percentile of |s|."""
    m = float(np.percentile(np.abs(s), q))
    if m <= 0:
        m = 1e-12
    return TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input_npz', type=str, required=True,
                   help='Path to attribution.npz produced by compute_saliency.py')
    p.add_argument('--target', type=str, default='norm', choices=['norm', 'pc1'])
    p.add_argument('--output', type=str, default=None,
                   help='Output PNG path. Defaults to <npz_dir>/saliency_<target>.png')
    p.add_argument('--clip_percentile', type=float, default=99.0,
                   help='Color scale uses ±this percentile of |s| for saturation.')
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input_npz)
    data = np.load(in_path, allow_pickle=True)

    target = args.target
    if target == 'pc1' and 's_flux_pc1' not in data.files:
        raise SystemExit(
            f'{in_path} has no PC1 attribution. Re-run compute_saliency.py '
            f'with --latents_npz to enable it, or use --target norm.'
        )

    t = data['times']
    flux = data['flux']
    flux_err = data['flux_err']
    s_flux = data[f's_flux_{target}']
    s_err = data[f's_err_{target}']
    w_gate = data['w_gate']
    L = t.size
    f_full = float(data[f'f_{target}_full'])
    f_base = float(data[f'f_{target}_baseline'])
    total = float(s_flux.sum() + s_err.sum())

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 1.2]})
    ax_flux, ax_err, ax_gate = axes

    norm_flux = _percentile_norm(s_flux, args.clip_percentile)
    norm_err = _percentile_norm(s_err, args.clip_percentile)

    # Top: flux trace colored by s_flux.
    lc1 = _colored_segments(ax_flux, t, flux, s_flux, cmap='RdBu_r', norm=norm_flux, lw=0.8)
    ax_flux.scatter(t, flux, c=s_flux, cmap='RdBu_r', norm=norm_flux, s=3, linewidths=0, zorder=3)
    ax_flux.set_xlim(t.min(), t.max())
    ax_flux.set_ylim(flux.min() - 0.05 * (flux.max() - flux.min() + 1e-12),
                     flux.max() + 0.05 * (flux.max() - flux.min() + 1e-12))
    ax_flux.set_ylabel('flux')
    ax_flux.set_title(
        f'Gaia {data["gaia_id"]}  |  TIC {int(data["tic_id"])}  |  sector {int(data["sector"])}  |  '
        f'target=f({target})  |  baseline={str(data["baseline_mode"])}  |  '
        f'Σs={total:.3e}  expected={f_full - f_base:.3e}'
    )
    cb1 = fig.colorbar(lc1, ax=ax_flux, pad=0.01)
    cb1.set_label(f's_flux  (signed, ±p{args.clip_percentile})')

    # Middle: flux_err trace colored by s_err.
    lc2 = _colored_segments(ax_err, t, flux_err, s_err, cmap='RdBu_r', norm=norm_err, lw=0.8)
    ax_err.scatter(t, flux_err, c=s_err, cmap='RdBu_r', norm=norm_err, s=3, linewidths=0, zorder=3)
    ax_err.set_xlim(t.min(), t.max())
    ax_err.set_ylim(0, max(flux_err.max() * 1.05, 1e-6))
    ax_err.set_ylabel('flux_err')
    cb2 = fig.colorbar(lc2, ax=ax_err, pad=0.01)
    cb2.set_label(f's_err  (signed, ±p{args.clip_percentile})')
    # Sanity inset: relative magnitude of err attribution vs flux attribution.
    ratio = np.sum(np.abs(s_err)) / max(np.sum(np.abs(s_flux)), 1e-12)
    ax_err.text(0.99, 0.95, f'Σ|s_err|/Σ|s_flux| = {ratio:.2g}',
                transform=ax_err.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Bottom: write-gate magnitude (mean sigmoid over hidden dims, both directions).
    # "How strongly does the encoder write step t into memory" — non-decaying,
    # so structurally visible across the whole sequence.
    ax_gate.plot(t, w_gate, color='k', lw=0.6)
    ax_gate.fill_between(t, 0, w_gate, color='0.85', alpha=0.7)
    ax_gate.set_ylabel('write gate z_t')
    ax_gate.set_xlabel('time (BJD - 2457000, days)')
    ax_gate.set_xlim(t.min(), t.max())
    lo = float(np.percentile(w_gate, 1))
    hi = float(np.percentile(w_gate, 99))
    pad = max(1e-6, 0.05 * (hi - lo))
    ax_gate.set_ylim(max(0.0, lo - pad), hi + pad)

    fig.tight_layout()

    out_path = Path(args.output) if args.output else in_path.parent / f'saliency_{target}.png'
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'[save] {out_path}')


if __name__ == '__main__':
    main()
