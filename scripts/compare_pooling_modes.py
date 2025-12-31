#!/usr/bin/env python
"""
Compare age prediction results across different pooling modes.

Loads k-fold prediction CSVs from different pooling modes and creates
comparison plots and metrics tables.

Usage:
    python scripts/compare_pooling_modes.py \
        --version v1 \
        --output_dir output/age_predictor/pooling_comparison_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_predictions(base_dir: Path, version: str, pooling_mode: str):
    """Load predictions CSV for a specific pooling mode."""
    csv_path = base_dir / f"kfold_{version}_{pooling_mode}" / "kfold_predictions.csv"
    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    return df


def compute_metrics(df: pd.DataFrame):
    """Compute age prediction metrics."""
    true = df['true_age_myr'].values
    pred = df['predicted_age_myr'].values
    
    mae = np.mean(np.abs(pred - true))
    median_ae = np.median(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    
    log_residuals = np.log10(pred) - np.log10(true)
    log_mae = np.mean(np.abs(log_residuals))
    log_bias = np.mean(log_residuals)
    log_scatter = np.std(log_residuals)
    
    corr = np.corrcoef(pred, true)[0, 1]
    
    # Fraction within factor of 2
    ratio = pred / true
    within_2x = np.mean((ratio >= 0.5) & (ratio <= 2.0))
    
    return {
        'mae_myr': mae,
        'median_ae_myr': median_ae,
        'rmse_myr': rmse,
        'log_mae_dex': log_mae,
        'log_bias_dex': log_bias,
        'log_scatter_dex': log_scatter,
        'correlation': corr,
        'within_2x_fraction': within_2x,
        'n_samples': len(true),
    }


def plot_comparison(results: dict, output_dir: Path):
    """Create comparison plots."""
    pooling_modes = list(results.keys())
    n_modes = len(pooling_modes)
    
    if n_modes == 0:
        print("No results to plot!")
        return
    
    # Color scheme
    colors = {'mean': '#1f77b4', 'multiscale': '#ff7f0e', 'final': '#2ca02c'}
    
    # =========================================================================
    # Figure 1: Side-by-side scatter plots
    # =========================================================================
    fig, axes = plt.subplots(1, n_modes, figsize=(5*n_modes, 5))
    if n_modes == 1:
        axes = [axes]
    
    for ax, mode in zip(axes, pooling_modes):
        df = results[mode]['df']
        metrics = results[mode]['metrics']
        
        true = df['true_age_myr'].values
        pred = df['predicted_age_myr'].values
        bprp0 = df['bprp0'].values
        
        scatter = ax.scatter(true, pred, c=bprp0, cmap='coolwarm', 
                            alpha=0.5, s=10, vmin=0.5, vmax=2.0)
        
        # Perfect line
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax.plot(lims, lims, 'k--', lw=2, label='Perfect')
        ax.plot(lims, [l*2 for l in lims], 'k:', alpha=0.4)
        ax.plot(lims, [l/2 for l in lims], 'k:', alpha=0.4)
        
        ax.set_xlabel('True Age (Myr)', fontsize=12)
        ax.set_ylabel('Predicted Age (Myr)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{mode.upper()} pooling\nr={metrics["correlation"]:.3f}, σ={metrics["log_scatter_dex"]:.3f} dex', 
                     fontsize=12)
        
        # Metrics text
        textstr = f'MAE={metrics["mae_myr"]:.0f} Myr\nWithin 2x: {metrics["within_2x_fraction"]*100:.1f}%'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.colorbar(scatter, ax=axes[-1], label='BP-RP$_0$')
    plt.tight_layout()
    plt.savefig(output_dir / 'pooling_scatter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter comparison to {output_dir / 'pooling_scatter_comparison.png'}")
    
    # =========================================================================
    # Figure 2: Bar chart comparison of metrics
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    metric_names = ['correlation', 'log_scatter_dex', 'mae_myr', 
                    'median_ae_myr', 'within_2x_fraction', 'log_bias_dex']
    metric_labels = ['Correlation (r)', 'Log Scatter (dex)', 'MAE (Myr)',
                     'Median AE (Myr)', 'Fraction within 2x', 'Log Bias (dex)']
    higher_better = [True, False, False, False, True, False]
    
    for ax, metric, label, hb in zip(axes.flat, metric_names, metric_labels, higher_better):
        values = [results[m]['metrics'][metric] for m in pooling_modes]
        bar_colors = [colors.get(m, '#333333') for m in pooling_modes]
        
        bars = ax.bar(pooling_modes, values, color=bar_colors, edgecolor='black', alpha=0.8)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        
        # Highlight best
        if hb:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}' if abs(val) < 10 else f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Pooling Mode Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'pooling_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison to {output_dir / 'pooling_metrics_comparison.png'}")
    
    # =========================================================================
    # Figure 3: Residuals comparison
    # =========================================================================
    fig, axes = plt.subplots(1, n_modes, figsize=(5*n_modes, 4))
    if n_modes == 1:
        axes = [axes]
    
    for ax, mode in zip(axes, pooling_modes):
        df = results[mode]['df']
        true = df['true_age_myr'].values
        pred = df['predicted_age_myr'].values
        
        log_residuals = np.log10(pred) - np.log10(true)
        
        ax.hist(log_residuals, bins=50, color=colors.get(mode, '#333333'), 
                alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(np.mean(log_residuals), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean={np.mean(log_residuals):.3f}')
        
        ax.set_xlabel('Log Residual (dex)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{mode.upper()}\nσ={np.std(log_residuals):.3f} dex', fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pooling_residuals_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved residuals comparison to {output_dir / 'pooling_residuals_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Compare pooling modes for age prediction')
    parser.add_argument('--version', type=str, required=True, help='Version tag (e.g., v1)')
    parser.add_argument('--base_dir', type=str, default='output/age_predictor',
                        help='Base directory containing kfold results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: base_dir/pooling_comparison_VERSION)')
    parser.add_argument('--pooling_modes', nargs='+', default=['mean', 'multiscale', 'final'],
                        help='Pooling modes to compare')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / f"pooling_comparison_{args.version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Comparing pooling modes for version: {args.version}")
    print(f"Looking in: {base_dir}")
    print(f"Output to: {output_dir}")
    
    # Load all results
    results = {}
    for mode in args.pooling_modes:
        print(f"\nLoading {mode} pooling results...")
        df = load_predictions(base_dir, args.version, mode)
        if df is not None:
            metrics = compute_metrics(df)
            results[mode] = {'df': df, 'metrics': metrics}
            print(f"  Loaded {len(df)} samples")
            print(f"  Correlation: {metrics['correlation']:.3f}")
            print(f"  Log scatter: {metrics['log_scatter_dex']:.3f} dex")
    
    if len(results) == 0:
        print("\nNo results found! Make sure to run kfold_age_inference.sh with different pooling modes first.")
        return
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("POOLING MODE COMPARISON")
    print("=" * 70)
    
    header = f"{'Metric':<20}"
    for mode in results.keys():
        header += f"{mode:>15}"
    print(header)
    print("-" * 70)
    
    metric_order = ['correlation', 'log_scatter_dex', 'mae_myr', 'median_ae_myr', 
                    'within_2x_fraction', 'log_bias_dex', 'n_samples']
    metric_formats = {
        'correlation': '.3f',
        'log_scatter_dex': '.3f',
        'mae_myr': '.1f',
        'median_ae_myr': '.1f',
        'within_2x_fraction': '.1%',
        'log_bias_dex': '.3f',
        'n_samples': 'd',
    }
    
    for metric in metric_order:
        row = f"{metric:<20}"
        for mode in results.keys():
            val = results[mode]['metrics'][metric]
            fmt = metric_formats.get(metric, '.3f')
            row += f"{val:>15{fmt}}"
        print(row)
    
    print("=" * 70)
    
    # Find best for key metrics
    print("\nBest pooling mode:")
    print(f"  Highest correlation: {max(results.keys(), key=lambda m: results[m]['metrics']['correlation'])}")
    print(f"  Lowest log scatter:  {min(results.keys(), key=lambda m: results[m]['metrics']['log_scatter_dex'])}")
    print(f"  Lowest MAE:          {min(results.keys(), key=lambda m: results[m]['metrics']['mae_myr'])}")
    
    # Create plots
    print("\nCreating comparison plots...")
    plot_comparison(results, output_dir)
    
    # Save metrics JSON
    metrics_dict = {mode: results[mode]['metrics'] for mode in results.keys()}
    with open(output_dir / 'pooling_comparison_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics to {output_dir / 'pooling_comparison_metrics.json'}")
    
    # Save combined CSV with all predictions
    combined_dfs = []
    for mode, data in results.items():
        df_copy = data['df'].copy()
        df_copy['pooling_mode'] = mode
        combined_dfs.append(df_copy)
    
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    combined_path = output_dir / 'all_pooling_predictions.csv'
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved combined predictions to {combined_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
