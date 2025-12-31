"""Compare age predictions: Latent Space Model vs Gyro Baseline (BPRP0 + Prot only).

This script runs both models using k-fold cross-validation and creates
comparison plots showing the improvement from using light curve latent space.

Can either:
1. Run fresh k-fold training for both models (default)
2. Load pre-computed predictions from CSV files (--load_from_csv)

Usage:
    # Run fresh training:
    python scripts/compare_age_models.py \
        --model_path output/simple_rnn/models/baseline_v15_masking_real_60e.pt \
        --output_dir output/age_predictor/comparison
    
    # Load from saved predictions (no retraining):
    python scripts/compare_age_models.py \
        --load_from_csv \
        --latent_csv output/age_predictor/kfold/kfold_predictions.csv \
        --gyro_csv output/age_predictor/gyro_baseline/gyro_kfold_predictions.csv \
        --output_dir output/age_predictor/comparison
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

# Ensure local 'src' is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kfold_age_inference import (
    load_data_fresh as load_latent_data,
    run_kfold_cv as run_latent_kfold,
    AgePredictor
)
from kfold_gyro_baseline import (
    load_gyro_data,
    run_kfold_cv as run_gyro_kfold,
    GyroAgePredictor
)


def compute_metrics(predictions, true_ages):
    """Compute standard metrics."""
    mae = np.mean(np.abs(predictions - true_ages))
    rmse = np.sqrt(np.mean((predictions - true_ages) ** 2))
    med_ae = np.median(np.abs(predictions - true_ages))
    corr = np.corrcoef(predictions, true_ages)[0, 1]
    
    log_residuals = np.log10(predictions) - np.log10(true_ages)
    log_mae = np.mean(np.abs(log_residuals))
    log_bias = np.mean(log_residuals)
    log_scatter = np.std(log_residuals)
    
    return {
        'mae_myr': mae,
        'rmse_myr': rmse,
        'median_ae_myr': med_ae,
        'correlation': corr,
        'log_mae_dex': log_mae,
        'log_bias_dex': log_bias,
        'log_scatter_dex': log_scatter,
    }


def plot_comparison(latent_results, gyro_results, output_dir):
    """Create comparison plots between the two models."""
    
    latent_pred, latent_ages, latent_bprp0, latent_tic = latent_results
    gyro_pred, gyro_ages, gyro_bprp0, gyro_prot, gyro_tic = gyro_results
    
    # Find common samples (by TIC ID)
    latent_tic_set = set(latent_tic)
    gyro_tic_set = set(gyro_tic)
    common_tics = latent_tic_set & gyro_tic_set
    
    print(f"\nLatent model samples: {len(latent_tic)}")
    print(f"Gyro baseline samples: {len(gyro_tic)}")
    print(f"Common samples: {len(common_tics)}")
    
    # Create index mappings for common samples
    latent_idx = {tic: i for i, tic in enumerate(latent_tic)}
    gyro_idx = {tic: i for i, tic in enumerate(gyro_tic)}
    
    common_list = list(common_tics)
    latent_common_idx = [latent_idx[tic] for tic in common_list]
    gyro_common_idx = [gyro_idx[tic] for tic in common_list]
    
    # Extract common sample data
    common_true_ages = latent_ages[latent_common_idx]
    common_latent_pred = latent_pred[latent_common_idx]
    common_gyro_pred = gyro_pred[gyro_common_idx]
    common_bprp0 = latent_bprp0[latent_common_idx]
    common_prot = gyro_prot[gyro_common_idx]
    
    # Compute metrics for common samples
    latent_metrics = compute_metrics(common_latent_pred, common_true_ages)
    gyro_metrics = compute_metrics(common_gyro_pred, common_true_ages)
    
    # Log residuals
    latent_log_res = np.log10(common_latent_pred) - np.log10(common_true_ages)
    gyro_log_res = np.log10(common_gyro_pred) - np.log10(common_true_ages)
    
    print(f"\n=== Comparison on {len(common_tics)} common samples ===")
    print(f"{'Metric':<20} {'Latent Model':<15} {'Gyro Baseline':<15} {'Improvement':<15}")
    print("-" * 65)
    for key in ['mae_myr', 'log_scatter_dex', 'correlation']:
        lat_val = latent_metrics[key]
        gyro_val = gyro_metrics[key]
        if key == 'correlation':
            improvement = (lat_val - gyro_val) / gyro_val * 100
            print(f"{key:<20} {lat_val:<15.3f} {gyro_val:<15.3f} {improvement:+.1f}%")
        else:
            improvement = (gyro_val - lat_val) / gyro_val * 100
            print(f"{key:<20} {lat_val:<15.3f} {gyro_val:<15.3f} {improvement:+.1f}%")
    
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Side-by-side scatter plots
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(common_true_ages, common_latent_pred, c=common_bprp0, cmap='coolwarm', 
                alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    min_age, max_age = common_true_ages.min(), common_true_ages.max()
    ax1.plot([min_age, max_age], [min_age, max_age], 'k--', lw=2)
    ax1.set_xlabel('True Age (Myr)', fontsize=12)
    ax1.set_ylabel('Predicted Age (Myr)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title(f'Latent Space Model (r={latent_metrics["correlation"]:.3f})', fontsize=14)
    textstr = f'σ = {latent_metrics["log_scatter_dex"]:.2f} dex'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2 = fig.add_subplot(2, 3, 2)
    scatter = ax2.scatter(common_true_ages, common_gyro_pred, c=common_bprp0, cmap='coolwarm', 
                          alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    ax2.plot([min_age, max_age], [min_age, max_age], 'k--', lw=2)
    ax2.set_xlabel('True Age (Myr)', fontsize=12)
    ax2.set_ylabel('Predicted Age (Myr)', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f'Gyro Baseline: BPRP0 + Prot (r={gyro_metrics["correlation"]:.3f})', fontsize=14)
    textstr = f'σ = {gyro_metrics["log_scatter_dex"]:.2f} dex'
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('BP-RP$_0$', fontsize=10)
    
    # 3. Direct comparison: latent vs gyro predictions
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(common_gyro_pred, common_latent_pred, c=np.log10(common_true_ages), 
                cmap='viridis', alpha=0.6, s=15)
    ax3.plot([min_age, max_age], [min_age, max_age], 'k--', lw=2)
    ax3.set_xlabel('Gyro Baseline Prediction (Myr)', fontsize=12)
    ax3.set_ylabel('Latent Model Prediction (Myr)', fontsize=12)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_title('Model Predictions Comparison', fontsize=14)
    
    # 4. Residual histograms (overlaid)
    ax4 = fig.add_subplot(2, 3, 4)
    bins = np.linspace(-1.5, 1.5, 61)
    ax4.hist(latent_log_res, bins=bins, alpha=0.7, label=f'Latent (σ={latent_metrics["log_scatter_dex"]:.2f})', 
             edgecolor='blue', color='blue')
    ax4.hist(gyro_log_res, bins=bins, alpha=0.5, label=f'Gyro (σ={gyro_metrics["log_scatter_dex"]:.2f})', 
             edgecolor='orange', color='orange')
    ax4.axvline(0, color='k', linestyle='--', lw=2)
    ax4.set_xlabel('Log Residual (dex)', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Residual Distributions', fontsize=14)
    ax4.legend()
    
    # 5. Improvement vs true age (binned)
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Bin by true age
    log_ages = np.log10(common_true_ages)
    n_bins = 10
    bin_edges = np.linspace(log_ages.min(), log_ages.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    latent_scatter_per_bin = []
    gyro_scatter_per_bin = []
    
    for i in range(n_bins):
        mask = (log_ages >= bin_edges[i]) & (log_ages < bin_edges[i+1])
        if mask.sum() > 10:
            latent_scatter_per_bin.append(np.std(latent_log_res[mask]))
            gyro_scatter_per_bin.append(np.std(gyro_log_res[mask]))
        else:
            latent_scatter_per_bin.append(np.nan)
            gyro_scatter_per_bin.append(np.nan)
    
    ax5.plot(10**bin_centers, latent_scatter_per_bin, 'o-', label='Latent Model', markersize=8)
    ax5.plot(10**bin_centers, gyro_scatter_per_bin, 's-', label='Gyro Baseline', markersize=8)
    ax5.set_xlabel('True Age (Myr)', fontsize=12)
    ax5.set_ylabel('Log Scatter (dex)', fontsize=12)
    ax5.set_xscale('log')
    ax5.set_title('Scatter vs Age (binned)', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Improvement vs BPRP0 (binned)
    ax6 = fig.add_subplot(2, 3, 6)
    
    bprp_bins = np.linspace(common_bprp0.min(), common_bprp0.max(), n_bins + 1)
    bprp_centers = (bprp_bins[:-1] + bprp_bins[1:]) / 2
    
    latent_scatter_per_bprp = []
    gyro_scatter_per_bprp = []
    
    for i in range(n_bins):
        mask = (common_bprp0 >= bprp_bins[i]) & (common_bprp0 < bprp_bins[i+1])
        if mask.sum() > 10:
            latent_scatter_per_bprp.append(np.std(latent_log_res[mask]))
            gyro_scatter_per_bprp.append(np.std(gyro_log_res[mask]))
        else:
            latent_scatter_per_bprp.append(np.nan)
            gyro_scatter_per_bprp.append(np.nan)
    
    ax6.plot(bprp_centers, latent_scatter_per_bprp, 'o-', label='Latent Model', markersize=8)
    ax6.plot(bprp_centers, gyro_scatter_per_bprp, 's-', label='Gyro Baseline', markersize=8)
    ax6.set_xlabel('BP-RP$_0$', fontsize=12)
    ax6.set_ylabel('Log Scatter (dex)', fontsize=12)
    ax6.set_title('Scatter vs Color (binned)', fontsize=14)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot to {output_dir / 'model_comparison.png'}")
    
    # Save combined predictions CSV
    combined_df = pd.DataFrame({
        'TIC_ID': common_list,
        'true_age_myr': common_true_ages,
        'latent_pred_myr': common_latent_pred,
        'gyro_pred_myr': common_gyro_pred,
        'bprp0': common_bprp0,
        'prot_days': common_prot,
        'latent_log_residual': latent_log_res,
        'gyro_log_residual': gyro_log_res,
    })
    combined_df.to_csv(output_dir / 'combined_predictions.csv', index=False)
    print(f"Saved combined predictions to {output_dir / 'combined_predictions.csv'}")
    
    # Save comparison metrics
    comparison_metrics = {
        'n_common_samples': len(common_tics),
        'latent_model': latent_metrics,
        'gyro_baseline': gyro_metrics,
        'improvement': {
            'log_scatter_reduction_pct': (gyro_metrics['log_scatter_dex'] - latent_metrics['log_scatter_dex']) / gyro_metrics['log_scatter_dex'] * 100,
            'mae_reduction_pct': (gyro_metrics['mae_myr'] - latent_metrics['mae_myr']) / gyro_metrics['mae_myr'] * 100,
            'correlation_improvement_pct': (latent_metrics['correlation'] - gyro_metrics['correlation']) / gyro_metrics['correlation'] * 100,
        }
    }
    with open(output_dir / 'comparison_metrics.json', 'w') as f:
        json.dump(comparison_metrics, f, indent=2)
    
    return comparison_metrics


def main():
    parser = argparse.ArgumentParser(description='Compare latent space vs gyro baseline age predictions')
    
    # Option to load from pre-computed CSV files
    parser.add_argument('--load_from_csv', action='store_true', 
                        help='Load predictions from existing CSV files instead of retraining')
    parser.add_argument('--latent_csv', type=str, default='output/age_predictor/kfold/kfold_predictions.csv',
                        help='Path to latent model predictions CSV (used with --load_from_csv)')
    parser.add_argument('--gyro_csv', type=str, default='output/age_predictor/gyro_baseline/gyro_kfold_predictions.csv',
                        help='Path to gyro baseline predictions CSV (used with --load_from_csv)')
    
    # Latent model settings (only needed if not loading from CSV)
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained BiDirectionalMinGRU model')
    parser.add_argument('--h5_path', type=str, default='data/timeseries.h5', help='Path to H5 data file')
    parser.add_argument('--pickle_path', type=str, default='data/star_sector_lc_formatted.pickle', help='Path to pickle file')
    parser.add_argument('--age_csv', type=str, default='data/TIC_cf_data.csv', help='Path to CSV with ages, BPRP0, Prot')
    parser.add_argument('--output_dir', type=str, default='output/age_predictor/comparison', help='Output directory')
    
    # RNN model settings
    parser.add_argument('--hidden_size', type=int, default=64, help='RNN hidden size')
    parser.add_argument('--direction', type=str, default='bi', choices=['forward', 'backward', 'bi'])
    parser.add_argument('--mode', type=str, default='parallel', choices=['sequential', 'parallel'])
    parser.add_argument('--use_flow', action='store_true', help='Whether RNN uses flow head')
    parser.add_argument('--max_length', type=int, default=2048, help='Max sequence length')
    parser.add_argument('--pooling_mode', type=str, default='mean', choices=['mean', 'multiscale', 'final'])
    
    # Data selection
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to use')
    parser.add_argument('--stratify_by_age', action='store_true', help='Stratified sampling by age')
    parser.add_argument('--n_age_bins', type=int, default=20, help='Number of age bins for stratification')
    parser.add_argument('--bprp0_min', type=float, default=None, help='Min BPRP0 cut')
    parser.add_argument('--bprp0_max', type=float, default=None, help='Max BPRP0 cut')
    
    # MLP settings (same for both models for fair comparison)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32], help='MLP hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.load_from_csv:
        # ========== Load from pre-computed CSV files ==========
        print("\n" + "="*60)
        print("LOADING FROM PRE-COMPUTED CSV FILES")
        print("="*60)
        
        # Load latent model predictions
        print(f"\nLoading latent model predictions from: {args.latent_csv}")
        latent_df = pd.read_csv(args.latent_csv)
        latent_pred = latent_df['predicted_age_myr'].values
        latent_ages = latent_df['true_age_myr'].values
        latent_bprp0 = latent_df['bprp0'].values
        latent_tics = latent_df['TIC_ID'].values
        print(f"  Loaded {len(latent_pred)} samples")
        
        # Load gyro baseline predictions
        print(f"\nLoading gyro baseline predictions from: {args.gyro_csv}")
        gyro_df = pd.read_csv(args.gyro_csv)
        gyro_pred = gyro_df['predicted_age_myr'].values
        gyro_ages = gyro_df['true_age_myr'].values
        gyro_bprp0 = gyro_df['bprp0'].values
        gyro_prot = gyro_df['prot_days'].values
        gyro_tics = gyro_df['TIC_ID'].values
        print(f"  Loaded {len(gyro_pred)} samples")
        
    else:
        # ========== Run fresh k-fold training ==========
        if args.model_path is None:
            parser.error("--model_path is required when not using --load_from_csv")
        
        # ========== Run Latent Space Model ==========
        print("\n" + "="*60)
        print("LATENT SPACE MODEL")
        print("="*60)
        
        latent_vectors, ages, bprp0, tic_ids = load_latent_data(
            model_path=args.model_path,
            h5_path=args.h5_path,
            pickle_path=args.pickle_path,
            csv_path=args.age_csv,
            hidden_size=args.hidden_size,
            direction=args.direction,
            mode=args.mode,
            use_flow=args.use_flow,
            max_length=args.max_length,
            batch_size=32,
            pooling_mode=args.pooling_mode,
            max_samples=args.max_samples,
            stratify_by_age=args.stratify_by_age,
            n_age_bins=args.n_age_bins,
            bprp0_min=args.bprp0_min,
            bprp0_max=args.bprp0_max,
            device=device,
        )
        
        latent_pred, latent_ages, latent_tics, _, _ = run_latent_kfold(
            latent_vectors, ages, bprp0, tic_ids,
            n_folds=args.n_folds,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
        latent_bprp0 = bprp0
        
        # ========== Run Gyro Baseline ==========
        print("\n" + "="*60)
        print("GYRO BASELINE (BPRP0 + Prot only)")
        print("="*60)
        
        gyro_bprp0, gyro_prot, gyro_ages, gyro_tics = load_gyro_data(
            args.age_csv,
            bprp0_min=args.bprp0_min,
            bprp0_max=args.bprp0_max
        )
        
        gyro_pred, gyro_ages, gyro_bprp0, gyro_prot, gyro_tics, _, _ = run_gyro_kfold(
            gyro_bprp0, gyro_prot, gyro_ages, gyro_tics,
            n_folds=args.n_folds,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
    
    # ========== Compare ==========
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    latent_results = (latent_pred, latent_ages, latent_bprp0, latent_tics)
    gyro_results = (gyro_pred, gyro_ages, gyro_bprp0, gyro_prot, gyro_tics)
    
    comparison_metrics = plot_comparison(latent_results, gyro_results, output_dir)
    
    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
