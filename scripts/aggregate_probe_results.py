"""Aggregate kfold_latent_probe.py outputs into one comparison table.

Reads every metrics.json under --root and emits a Markdown table plus a tidy
CSV. Rows = probes; columns = baselines (none / gaussian / shuffle), each
showing Pearson r and MAE in transformed space.

Usage:
    python scripts/aggregate_probe_results.py \\
        --root output/latent_probes \\
        --out_md  output/latent_probes/summary.md \\
        --out_csv output/latent_probes/summary.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

BASELINES = ['none', 'MLP', 'gaussian', 'shuffle']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='output/latent_probes')
    ap.add_argument('--out_md',  default='output/latent_probes/summary.md')
    ap.add_argument('--out_csv', default='output/latent_probes/summary.csv')
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for mj in sorted(root.glob('*/metrics.json')):
        d = json.loads(mj.read_text())
        rows.append({
            'probe':     d['probe'],
            'baseline':  d['baseline'],
            'transform': d.get('transform', ''),
            'n':         d['n'],
            'r':         d['pearson_r_transformed'],
            'mae':       d['mae_transformed'],
            'rmse':      d['rmse_transformed'],
            'r_raw':     d.get('pearson_r_raw'),
            'mae_raw':   d.get('mae_raw'),
            'run_dir':   str(mj.parent.relative_to(root)),
        })
    if not rows:
        print(f'no metrics.json under {root}')
        return

    long = pd.DataFrame(rows).sort_values(['probe', 'baseline'])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out_csv, index=False)
    print(f'wrote {args.out_csv}  ({len(long)} rows)')

    # Wide table: one row per probe, columns per baseline
    pivot_r   = long.pivot_table(index='probe', columns='baseline', values='r')
    pivot_mae = long.pivot_table(index='probe', columns='baseline', values='mae')
    pivot_n   = long.pivot_table(index='probe', columns='baseline', values='n')
    transform_by_probe = long.groupby('probe')['transform'].first()

    cols_present = [b for b in BASELINES if b in pivot_r.columns]
    lines = ['# Latent-space probe summary', '',
             'Pearson r and MAE in transformed space (asinh for flux moments, '
             'log10 for Prot). Baseline columns: `none` = real RNN latents '
             '(BiDirectionalMinGRU multiscale-pooled); `MLP` = pooled MLP-baseline '
             'features (k-averaged, mean+std); `gaussian` = iid N(0,1) noise of the '
             'same shape; `shuffle` = real latents permuted row-wise. The MLP cache '
             'covers all sequences in the H5 files, the RNN cache covers a subset, '
             'hence the per-baseline `n` columns differ.', '',
             '| probe | transform | ' +
             ' | '.join(f'n ({b})' for b in cols_present) + ' | ' +
             ' | '.join(f'r ({b})' for b in cols_present) + ' | ' +
             ' | '.join(f'MAE ({b})' for b in cols_present) + ' |',
             '|' + '---|' * (2 + 3 * len(cols_present))]
    for probe in pivot_r.index:
        tr = transform_by_probe.loc[probe]
        n_cells   = [f'{int(pivot_n.loc[probe, b])}'
                     if b in pivot_n.columns and pd.notna(pivot_n.loc[probe, b]) else '—'
                     for b in cols_present]
        r_cells   = [f'{pivot_r.loc[probe, b]:+.3f}'
                     if b in pivot_r.columns and pd.notna(pivot_r.loc[probe, b]) else '—'
                     for b in cols_present]
        mae_cells = [f'{pivot_mae.loc[probe, b]:.4f}'
                     if b in pivot_mae.columns and pd.notna(pivot_mae.loc[probe, b]) else '—'
                     for b in cols_present]
        lines.append(f'| {probe} | {tr} | ' +
                     ' | '.join(n_cells) + ' | ' +
                     ' | '.join(r_cells) + ' | ' +
                     ' | '.join(mae_cells) + ' |')

    Path(args.out_md).write_text('\n'.join(lines) + '\n')
    print(f'wrote {args.out_md}')
    print()
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
