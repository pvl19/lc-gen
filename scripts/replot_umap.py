#!/usr/bin/env python
"""
Replot UMAP embeddings from saved data without recomputing UMAP.

This script loads pre-computed UMAP embeddings from .npz files created by
plot_umap_latent.py and allows you to tweak plot settings (colormap, point size,
alpha, age range, etc.) without re-running the expensive UMAP computation.

Usage:
    python scripts/replot_umap.py \
        --data_file output/simple_rnn/umap_plots/umap_age_baseline_v17_multiscale_data.npz \
        --output output/simple_rnn/umap_plots/umap_replot_v1.png

    # With custom settings
    python scripts/replot_umap.py \
        --data_file output/simple_rnn/umap_plots/umap_age_baseline_v17_multiscale_data.npz \
        --output output/simple_rnn/umap_plots/umap_custom.png \
        --cmap viridis \
        --point_size 20 \
        --alpha 0.7 \
        --age_min 10 \
        --age_max 1000 \
        --figsize 10 10
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def load_umap_data(data_file: str):
    """Load saved UMAP embedding and associated data.
    
    Args:
        data_file: path to .npz file from plot_umap_latent.py
        
    Returns:
        embedding: (N, 2) UMAP coordinates
        ages: (N,) ages in Myr
        bprp0: (N,) BPRP0 values (may be empty array if not saved)
        latent_vectors: (N, D) original latent vectors (optional, may not exist)
    """
    data = np.load(data_file)
    embedding = data['embedding']
    ages = data['ages']
    
    # bprp0 may or may not be present
    bprp0 = data.get('bprp0', None)
    if bprp0 is not None and len(bprp0) == 0:
        bprp0 = None
    
    # latent_vectors may or may not be present
    latent_vectors = data.get('latent_vectors', None)
    
    print(f"Loaded UMAP data from {data_file}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Age range: {ages.min():.1f} - {ages.max():.1f} Myr")
    print(f"  N samples: {len(ages)}")
    if bprp0 is not None:
        print(f"  BPRP0 range: {bprp0.min():.2f} - {bprp0.max():.2f}")
    
    return embedding, ages, bprp0, latent_vectors


def replot_umap(embedding, ages, output_path,
                n_samples=None, bprp0_min=None, bprp0_max=None, bprp0=None):
    """Create UMAP plot with labels for sample info.
    
    Args:
        embedding: (N, 2) UMAP coordinates
        ages: (N,) ages in Myr
        output_path: where to save the plot
        n_samples: number of samples (for labeling, uses len(ages) if None)
        bprp0_min: min BPRP0 for filtering (if bprp0 array provided) or labeling
        bprp0_max: max BPRP0 for filtering (if bprp0 array provided) or labeling
        bprp0: (N,) BPRP0 values for filtering (optional)
    """
    # Apply BPRP0 filtering if bprp0 array is provided
    if bprp0 is not None and (bprp0_min is not None or bprp0_max is not None):
        mask = np.ones(len(ages), dtype=bool)
        if bprp0_min is not None:
            mask &= (bprp0 >= bprp0_min)
        if bprp0_max is not None:
            mask &= (bprp0 <= bprp0_max)
        
        embedding = embedding[mask]
        ages = ages[mask]
        print(f"Filtered to {len(ages)} samples within BPRP0 range")
    
    # Plot settings (hardcoded for consistency)
    cmap = 'viridis'
    point_size = 15
    alpha = 0.6
    figsize = (10, 6)
    dpi = 300
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        embedding[:, 0], 
        embedding[:, 1],
        c=ages,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        norm=LogNorm(vmin=ages.min(), vmax=ages.max()),
        rasterized=True
    )
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Age (Myr)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    ax.set_xlabel('UMAP 1', fontsize=16)
    ax.set_ylabel('UMAP 2', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('', fontsize=14)
    
    # Stats text box
    n_display = n_samples if n_samples is not None else len(ages)
    textstr_lines = [
    ]
    
    # Add BPRP0 range if provided
    if bprp0_min is not None or bprp0_max is not None:
        bprp_str = f'G$_{{(BP-RP)_0}}$: {bprp0_min or ""} - {bprp0_max or ""}'
        textstr_lines.append(bprp_str)
    
    textstr = '\n'.join(textstr_lines)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Replot UMAP embeddings from saved data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/replot_umap.py --data_file output/.../umap_data.npz --output new_plot.png
  python scripts/replot_umap.py --data_file data.npz --output plot.png --n_samples 5000 --bprp0_min 0.5 --bprp0_max 1.5
        """
    )
    
    # Required
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to .npz file from plot_umap_latent.py')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for the plot')
    
    # Labeling parameters
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples (for labeling)')
    parser.add_argument('--bprp0_min', type=float, default=None,
                        help='Min BPRP0 for filtering/labeling')
    parser.add_argument('--bprp0_max', type=float, default=None,
                        help='Max BPRP0 for filtering/labeling')
    
    args = parser.parse_args()
    
    # Load data
    embedding, ages, bprp0, _ = load_umap_data(args.data_file)
    
    # Replot
    replot_umap(
        embedding, ages, args.output,
        n_samples=args.n_samples,
        bprp0_min=args.bprp0_min,
        bprp0_max=args.bprp0_max,
        bprp0=bprp0,
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
