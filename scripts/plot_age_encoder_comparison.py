"""Compare RNN-pooled vs MLP-pooled age recovery.

Reads kfold metrics + predictions from the two age-inference output dirs and
writes a comparison summary CSV plus a 2-panel scatter plot (true vs predicted
log10 age, one panel per encoder family).

See docs/plans/2026-05-12_mlp-pooled-age-inference.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = [
    {
        'label': 'RNN-pooled (BiDirectionalMinGRU, multiscale)',
        'short': 'rnn_pooled',
        'dir': 'final_model/parallel_fixed/e110/age-inference-cfonly-multiscale-latent_max',
    },
    {
        'label': 'MLP-pooled (local-window, k\u2208{1,8,64,720}, mean+std)',
        'short': 'mlp_pooled',
        'dir': 'output/baseline_comparison/age_inference_mlp_pooled',
    },
]


def load_run(d: Path) -> dict:
    metrics = json.loads((d / 'kfold_metrics.json').read_text())
    preds = pd.read_csv(d / 'kfold_predictions.csv')
    return {'metrics': metrics, 'preds': preds}


def write_summary_csv(rows, out_path: Path) -> None:
    cols = ['encoder', 'short', 'n_samples', 'mae_dex', 'median_ae_dex',
            'rmse_dex', 'bias_dex', 'scatter_dex', 'correlation']
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_path, index=False)
    print(f'[write] {out_path}')
    print(df.to_string(index=False))


def plot_comparison(runs, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    age_min, age_max = np.inf, -np.inf
    for r in runs:
        p = r['data']['preds']
        age_min = min(age_min, p['log10_true_age'].min(), p['log10_pred_median'].min())
        age_max = max(age_max, p['log10_true_age'].max(), p['log10_pred_median'].max())
    pad = 0.05 * (age_max - age_min)
    lo, hi = age_min - pad, age_max + pad

    for ax, r in zip(axes, runs):
        p = r['data']['preds']
        m = r['data']['metrics']
        ax.scatter(p['log10_true_age'], p['log10_pred_median'],
                   s=6, alpha=0.35, c='C0', edgecolors='none')
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$\log_{10}$(true age / Myr)')
        ax.set_ylabel(r'$\log_{10}$(predicted age / Myr)')
        title = (f"{r['label']}\n"
                 f"N={m['n_samples']}, r={m['correlation']:.3f}, "
                 f"MAE={m['mae_dex']:.3f} dex")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)

    fig.suptitle('Age recovery — encoder family comparison '
                 '(10-fold CV, ChronoFlow subset)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[write] {out_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', default='output/baseline_comparison',
                   help='Where to write summary CSV and PNG.')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in RUNS:
        d = Path(run['dir'])
        data = load_run(d)
        run['data'] = data
        m = data['metrics']
        rows.append({
            'encoder': run['label'],
            'short': run['short'],
            'n_samples': m['n_samples'],
            'mae_dex': m['mae_dex'],
            'median_ae_dex': m['median_ae_dex'],
            'rmse_dex': m['rmse_dex'],
            'bias_dex': m['bias_dex'],
            'scatter_dex': m['scatter_dex'],
            'correlation': m['correlation'],
        })

    write_summary_csv(rows, out_dir / 'age_encoder_comparison.csv')
    plot_comparison(RUNS, out_dir / 'age_encoder_comparison.png')


if __name__ == '__main__':
    main()
