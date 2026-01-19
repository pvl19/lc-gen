#!/usr/bin/env python3
"""
Replot residual histograms from saved combined_predictions.csv without rerunning
the full comparison pipeline. Edit parameters below for publication styling.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============== PLOT CONFIGURATION ==============
# Histogram bins
BIN_MIN = -1.5
BIN_MAX = 1.5
N_BINS = 61

# Colors and styling
LATENT_COLOR = 'blue'
GYRO_COLOR = 'orange'
LATENT_ALPHA = 0.7
GYRO_ALPHA = 0.5
LATENT_LABEL = 'Latent Space'
GYRO_LABEL = r'P$_{rot}$ Baseline'

# Font sizes
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10

# Figure size
FIGWIDTH = 6
FIGHEIGHT = 6
DPI = 150

# Plot text
TITLE = ''
XLABEL = r'Residual (log Myr)'
YLABEL = 'Density'

# Options
SHOW_SIGMA = True
SHOW_GRID = True
GRID_ALPHA = 0.3
# ================================================


def main():
    parser = argparse.ArgumentParser(description='Replot residual histograms from saved predictions')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to combined_predictions.csv from compare_age_models.py')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for the plot (e.g., output/residuals.png)')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Get residuals (either from saved columns or compute from predictions)
    if 'latent_log_residual' in df.columns and 'gyro_log_residual' in df.columns:
        latent_log_res = df['latent_log_residual'].values
        gyro_log_res = df['gyro_log_residual'].values
    else:
        # Compute residuals from predictions
        latent_log_res = np.log10(df['latent_pred_myr'].values) - np.log10(df['true_age_myr'].values)
        gyro_log_res = np.log10(df['gyro_pred_myr'].values) - np.log10(df['true_age_myr'].values)
    
    # Compute scatter metrics
    latent_scatter = np.std(latent_log_res)
    gyro_scatter = np.std(gyro_log_res)
    
    print(f"Latent log scatter: {latent_scatter:.3f} dex")
    print(f"Gyro log scatter: {gyro_scatter:.3f} dex")
    
    # Create labels
    if SHOW_SIGMA:
        latent_label = f'{LATENT_LABEL} (σ={latent_scatter:.2f})'
        gyro_label = f'{GYRO_LABEL} (σ={gyro_scatter:.2f})'
    else:
        latent_label = LATENT_LABEL
        gyro_label = GYRO_LABEL
    
    # Create figure
    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))
    
    # Create bins
    bins = np.linspace(BIN_MIN, BIN_MAX, N_BINS)
    
    # Plot histograms
    ax.hist(latent_log_res, bins=bins, alpha=LATENT_ALPHA, 
            label=latent_label, edgecolor=LATENT_COLOR, color=LATENT_COLOR, density=True)
    ax.hist(gyro_log_res, bins=bins, alpha=GYRO_ALPHA, 
            label=gyro_label, edgecolor=GYRO_COLOR, color=GYRO_COLOR, density=True)
    
    # Styling
    ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE)
    # ax.set_ylabel(YLABEL, fontsize=LABEL_FONTSIZE)
    # if TITLE:
    #     ax.set_title(TITLE, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', length=0)  # Hide y-axis tick marks but keep gridlines
    ax.set_yticklabels([])  # Hide y-axis tick labels
    ax.legend(fontsize=LEGEND_FONTSIZE)
    
    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA)
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_path}")


if __name__ == '__main__':
    main()
