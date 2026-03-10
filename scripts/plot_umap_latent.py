"""UMAP visualization of latent space colored by stellar age.

This script:
1. Loads a trained BiDirectionalMinGRU model
2. Extracts latent representations (hidden states) for all light curves
3. Loads stellar ages from TIC_cf_data.csv matched via TIC ID
4. Creates a UMAP embedding colored by stellar age
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Ensure local 'src' is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lcgen.models.simple_min_gru import BiDirectionalMinGRU
from lcgen.models.MetadataAgePredictor import DEFAULT_METADATA_FIELDS


def get_stratified_indices(ages: np.ndarray, max_samples: int, n_bins: int = 20, seed: int = 42) -> np.ndarray:
    """Get indices for stratified sampling across the age distribution.
    
    Samples roughly equally from logarithmic age bins to ensure good coverage
    of the full age range, avoiding over-representation of cluster stars.
    
    Args:
        ages: array of ages (may contain NaN)
        max_samples: target number of samples
        n_bins: number of age bins to stratify across
        seed: random seed for reproducibility
        
    Returns:
        Array of indices to use for sampling
    """
    np.random.seed(seed)
    
    # Only consider samples with valid ages
    valid_mask = ~np.isnan(ages)
    valid_indices = np.where(valid_mask)[0]
    valid_ages = ages[valid_mask]
    
    if len(valid_indices) == 0:
        raise ValueError("No samples with valid ages found")
    
    if max_samples >= len(valid_indices):
        print(f'Requested {max_samples} samples but only {len(valid_indices)} have valid ages. Using all.')
        return valid_indices
    
    # Use log-spaced bins since ages span orders of magnitude
    log_ages = np.log10(valid_ages)
    bin_edges = np.linspace(log_ages.min(), log_ages.max() + 1e-10, n_bins + 1)
    bin_indices = np.digitize(log_ages, bin_edges) - 1  # 0 to n_bins-1
    
    # Group indices by bin
    bin_to_indices = {i: [] for i in range(n_bins)}
    for idx, bin_idx in zip(valid_indices, bin_indices):
        bin_to_indices[bin_idx].append(idx)
    
    # Initial allocation: equal samples per bin
    samples_per_bin = max_samples // n_bins
    
    selected_indices = []
    remaining_pool = []  # Indices from bins that have extra samples
    
    # First pass: take up to samples_per_bin from each bin
    for bin_idx in range(n_bins):
        bin_pool = bin_to_indices[bin_idx]
        
        if len(bin_pool) == 0:
            continue
        elif len(bin_pool) <= samples_per_bin:
            # Take all from this sparse bin
            selected_indices.extend(bin_pool)
        else:
            # Sample from this bin, keep rest in pool for potential redistribution
            np.random.shuffle(bin_pool)
            selected_indices.extend(bin_pool[:samples_per_bin])
            remaining_pool.extend(bin_pool[samples_per_bin:])
    
    # Second pass: fill up to max_samples from the remaining pool
    shortfall = max_samples - len(selected_indices)
    if shortfall > 0 and len(remaining_pool) > 0:
        np.random.shuffle(remaining_pool)
        selected_indices.extend(remaining_pool[:shortfall])
    
    selected_indices = np.array(selected_indices)
    
    # Report sampling distribution
    print(f'Stratified sampling: {len(selected_indices)} samples across {n_bins} log-age bins')
    log_age_range = (10**bin_edges[0], 10**bin_edges[-1])
    print(f'  Age range: {log_age_range[0]:.1f} - {log_age_range[1]:.1f} Myr')
    
    return selected_indices


def load_model(model_path: str, hidden_size: int, direction: str, mode: str, use_flow: bool, device: torch.device, num_meta_features: int = 0, use_conv_channels: bool = False, conv_config: dict = None):
    """Load trained model from checkpoint.

    Infers the architecture from the checkpoint (either from a stored
    'num_meta_features' key, or by inspecting the state dict weights) and
    raises ValueError if it does not match the requested num_meta_features.
    """
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
        # Determine num_meta_features from checkpoint
        if 'num_meta_features' in ckpt:
            ckpt_num_meta = ckpt['num_meta_features']
        elif 'meta_encoder.0.weight' in sd:
            # Infer from first linear layer: weight shape is (out, in) = (64, num_meta_features)
            ckpt_num_meta = sd['meta_encoder.0.weight'].shape[1]
        else:
            ckpt_num_meta = 0
    else:
        sd = ckpt
        ckpt_num_meta = num_meta_features  # old-format checkpoint, no way to verify

    if ckpt_num_meta != num_meta_features:
        raise ValueError(
            f"Architecture mismatch: checkpoint at '{model_path}' was trained with "
            f"num_meta_features={ckpt_num_meta}, but script requested "
            f"num_meta_features={num_meta_features}. "
            f"Pass --use_metadata (or omit it) to match the checkpoint."
        )

    model = BiDirectionalMinGRU(
        hidden_size=hidden_size,
        direction=direction,
        mode=mode,
        use_flow=use_flow,
        num_meta_features=num_meta_features,
        use_conv_channels=use_conv_channels,
        conv_config=conv_config
    ).to(device)

    model.load_state_dict(sd)
    model.eval()
    print(f'Loaded model from {model_path} (num_meta_features={num_meta_features})')
    return model


def compute_multiscale_features(h_valid: torch.Tensor, n_segments: int = 4) -> torch.Tensor:
    """Compute multi-scale temporal features from hidden states.
    
    Args:
        h_valid: (L, H) hidden states for valid timesteps
        n_segments: number of temporal segments for quartile pooling
        
    Returns:
        Feature vector combining multiple aggregation strategies
    """
    L, H = h_valid.shape
    features = []
    
    # 1. Global statistics
    global_mean = h_valid.mean(dim=0)  # (H,)
    global_std = h_valid.std(dim=0)    # (H,)
    global_max = h_valid.max(dim=0).values  # (H,)
    global_min = h_valid.min(dim=0).values  # (H,)
    features.extend([global_mean, global_std, global_max, global_min])
    
    # 2. Temporal segment pooling (quartiles)
    # Split sequence into n_segments and compute mean for each
    segment_size = max(1, L // n_segments)
    for seg_idx in range(n_segments):
        start = seg_idx * segment_size
        end = min((seg_idx + 1) * segment_size, L) if seg_idx < n_segments - 1 else L
        if start < L:
            seg_mean = h_valid[start:end, :].mean(dim=0)
            features.append(seg_mean)
        else:
            # If sequence is shorter than expected, pad with zeros
            features.append(torch.zeros(H, device=h_valid.device))
    
    # 3. First and last hidden states (capture beginning/end of light curve)
    features.append(h_valid[0, :])   # first
    features.append(h_valid[-1, :])  # last
    
    # 4. Temporal derivative statistics (how hidden states change over time)
    if L > 1:
        h_diff = h_valid[1:, :] - h_valid[:-1, :]  # (L-1, H)
        diff_mean = h_diff.mean(dim=0)
        diff_std = h_diff.std(dim=0)
        features.extend([diff_mean, diff_std])
    else:
        features.extend([torch.zeros(H, device=h_valid.device), torch.zeros(H, device=h_valid.device)])
    
    return torch.cat(features, dim=0)


def extract_latent_vectors(model, h5_path: str, device: torch.device, max_length: int = 16384,
                           batch_size: int = 32, pooling_mode: str = 'multiscale',
                           sample_indices: np.ndarray = None, use_metadata: bool = False,
                           use_conv_channels: bool = False):
    """Extract latent vectors for all light curves in the H5 file.

    Args:
        model: trained BiDirectionalMinGRU model
        h5_path: path to H5 data file
        device: torch device
        max_length: maximum sequence length
        batch_size: batch size for inference
        pooling_mode: how to aggregate hidden states
            - 'mean': simple mean pooling (original behavior)
            - 'multiscale': multi-scale temporal features (recommended)
            - 'final': use only final hidden states
        sample_indices: specific indices to process (None = all)
        use_metadata: whether to load and use stellar metadata
        use_conv_channels: whether to load and use conv channel data (power, ACF, f-statistic)

    Returns:
        Latent vectors array of shape (N, D) where D depends on pooling_mode
    """
    # Only load metadata if the model actually has a meta_encoder
    load_metadata = use_metadata and (model.meta_encoder is not None)
    metadata_features = DEFAULT_METADATA_FIELDS

    with h5py.File(h5_path, 'r') as f:
        # Get the actual size of the HDF5 dataset
        h5_size = len(f['flux'])

        if sample_indices is not None:
            # Filter indices to only include valid HDF5 indices (defensive check)
            valid_mask = sample_indices < h5_size
            if not np.all(valid_mask):
                n_invalid = np.sum(~valid_mask)
                print(f'Warning: {n_invalid} sample indices exceed HDF5 size ({h5_size}). Filtering to valid indices.')
                sample_indices = sample_indices[valid_mask]

            if len(sample_indices) == 0:
                raise ValueError('No valid sample indices remain after filtering')

            # Sort indices for efficient HDF5 access
            sorted_indices = np.sort(sample_indices)
            flux_all = f['flux'][sorted_indices]
            flux_err_all = f['flux_err'][sorted_indices]
            time_all = f['time'][sorted_indices]
            lengths = f['length'][sorted_indices]

            # Load conv channel data if requested
            if use_conv_channels:
                power_all = f['power'][sorted_indices]
                acf_all = f['acf'][sorted_indices]
                f_stat_all = f['f_stat'][sorted_indices]
            else:
                power_all = None
                acf_all = None
                f_stat_all = None

            # Load metadata only if the model has a meta_encoder
            if load_metadata and 'metadata' in f:
                meta_grp = f['metadata']
                meta_list = []
                for feat in metadata_features:
                    if feat in meta_grp:
                        vals = meta_grp[feat][sorted_indices]
                        if vals.dtype.kind == 'S':
                            vals = vals.astype(float)
                        meta_list.append(vals.astype(np.float32))
                    else:
                        meta_list.append(np.zeros(len(sorted_indices), dtype=np.float32))
                metadata_all = np.stack(meta_list, axis=1)
                metadata_all = np.nan_to_num(metadata_all, nan=0.0)
            else:
                metadata_all = None
        else:
            flux_all = f['flux'][:]
            flux_err_all = f['flux_err'][:]
            time_all = f['time'][:]
            lengths = f['length'][:]

            # Load conv channel data if requested
            if use_conv_channels:
                power_all = f['power'][:]
                acf_all = f['acf'][:]
                f_stat_all = f['f_stat'][:]
            else:
                power_all = None
                acf_all = None
                f_stat_all = None

            # Load metadata only if the model has a meta_encoder
            if load_metadata and 'metadata' in f:
                meta_grp = f['metadata']
                meta_list = []
                for feat in metadata_features:
                    if feat in meta_grp:
                        vals = meta_grp[feat][:]
                        if vals.dtype.kind == 'S':
                            vals = vals.astype(float)
                        meta_list.append(vals.astype(np.float32))
                    else:
                        meta_list.append(np.zeros(len(flux_all), dtype=np.float32))
                metadata_all = np.stack(meta_list, axis=1)
                metadata_all = np.nan_to_num(metadata_all, nan=0.0)
            else:
                metadata_all = None
    
    n_samples = len(lengths)
    latent_vectors = []
    
    print(f'Extracting latent vectors for {n_samples} light curves (pooling={pooling_mode})...')
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = range(start_idx, end_idx)
            
            # Get batch data
            flux_batch = torch.tensor(flux_all[start_idx:end_idx], dtype=torch.float32, device=device)
            flux_err_batch = torch.tensor(flux_err_all[start_idx:end_idx], dtype=torch.float32, device=device)
            time_batch = torch.tensor(time_all[start_idx:end_idx], dtype=torch.float32, device=device)
            lengths_batch = lengths[start_idx:end_idx]
            
            # Truncate to max_length for efficiency (model expects reasonable lengths)
            actual_max = min(max_length, flux_batch.shape[1])
            flux_batch = flux_batch[:, :actual_max]
            flux_err_batch = flux_err_batch[:, :actual_max]
            time_batch = time_batch[:, :actual_max]
            lengths_batch = np.minimum(lengths_batch, actual_max)
            
            # Create mask (1 = valid, 0 = padded)
            B, L = flux_batch.shape
            mask = torch.zeros(B, L, device=device)
            for i, length in enumerate(lengths_batch):
                mask[i, :length] = 1.0
            
            # Build input
            x_in = torch.stack([flux_batch, flux_err_batch], dim=-1)  # (B, L, 2)
            t_in = time_batch.unsqueeze(-1)  # (B, L, 1)
            
            # Build metadata tensor if available
            if metadata_all is not None:
                meta_batch = torch.tensor(metadata_all[start_idx:end_idx], dtype=torch.float32, device=device)
            else:
                meta_batch = None

            # Build conv data dict if available
            if use_conv_channels and power_all is not None:
                power_batch = torch.tensor(power_all[start_idx:end_idx], dtype=torch.float32, device=device)
                acf_batch = torch.tensor(acf_all[start_idx:end_idx], dtype=torch.float32, device=device)
                f_stat_batch = torch.tensor(f_stat_all[start_idx:end_idx], dtype=torch.float32, device=device)
                # Pass as dictionary (model expects dict format)
                conv_data_batch = {
                    'power': power_batch,
                    'acf': acf_batch,
                    'f_stat': f_stat_batch
                }
            else:
                conv_data_batch = None

            # Forward pass with return_states=True to get hidden states
            out = model(x_in, t_in, mask=mask, metadata=meta_batch, conv_data=conv_data_batch, return_states=True)
            
            # Get hidden states
            h_fwd = out.get('h_fwd_tensor')  # (B, L, H)
            h_bwd = out.get('h_bwd_tensor')  # (B, L, H)
            
            # Compute latent vector based on pooling mode
            for i in range(B):
                valid_len = lengths_batch[i]
                
                # Combine forward and backward hidden states
                if h_fwd is not None and h_bwd is not None:
                    h_combined = torch.cat([h_fwd[i, :valid_len, :], h_bwd[i, :valid_len, :]], dim=-1)
                elif h_fwd is not None:
                    h_combined = h_fwd[i, :valid_len, :]
                elif h_bwd is not None:
                    h_combined = h_bwd[i, :valid_len, :]
                else:
                    raise ValueError("No hidden states returned by model")
                
                # Apply pooling strategy
                if pooling_mode == 'mean':
                    latent = h_combined.mean(dim=0)
                elif pooling_mode == 'final':
                    latent = h_combined[-1, :]
                elif pooling_mode == 'multiscale':
                    latent = compute_multiscale_features(h_combined, n_segments=4)
                else:
                    raise ValueError(f"Unknown pooling_mode: {pooling_mode}")
                
                latent_vectors.append(latent.cpu().numpy())
            
            if (start_idx // batch_size) % 10 == 0:
                print(f'  Processed {end_idx}/{n_samples} light curves')
    
    latent_vectors = np.stack(latent_vectors, axis=0)
    print(f'Extracted latent vectors with shape: {latent_vectors.shape}')
    return latent_vectors


def load_ages(pickle_path: str, csv_path: str, sample_indices: np.ndarray = None):
    """Load ages, BPRP0, and Prot from CSV and map to light curves via TIC ID.

    Args:
        pickle_path: path to pickle file with metadata
        csv_path: path to CSV with stellar ages, BPRP0, and Prot
        sample_indices: specific indices to load (None = all)

    Returns:
        ages: array of ages (NaN for light curves without age data)
        bprp0: array of BPRP0 values (NaN for light curves without data)
        tic_ids: array of TIC IDs for each light curve
        prot: array of rotation periods in days (NaN for light curves without data)
    """
    # Load pickle to get TIC IDs for each light curve
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)

    metadatas = pickle_data['metadatas']
    if sample_indices is not None:
        sorted_indices = np.sort(sample_indices)
        metadatas = [metadatas[i] for i in sorted_indices]
    tic_ids = np.array([m['tic'] for m in metadatas])

    # Load age CSV
    age_df = pd.read_csv(csv_path)
    tic_to_age = dict(zip(age_df['TIC_ID'], age_df['age_Myr']))
    tic_to_bprp0 = dict(zip(age_df['TIC_ID'], age_df['BPRP0']))
    tic_to_prot = dict(zip(age_df['TIC_ID'], age_df.get('Prot', pd.Series([np.nan]*len(age_df)))))

    # Map ages, BPRP0, and Prot to each light curve
    ages = np.array([tic_to_age.get(tic, np.nan) for tic in tic_ids])
    bprp0 = np.array([tic_to_bprp0.get(tic, np.nan) for tic in tic_ids])
    prot = np.array([tic_to_prot.get(tic, np.nan) for tic in tic_ids])

    n_with_age = np.sum(~np.isnan(ages))
    print(f'Loaded ages for {n_with_age}/{len(ages)} light curves ({100*n_with_age/len(ages):.1f}%)')

    return ages, bprp0, tic_ids, prot


def plot_umap(latent_vectors: np.ndarray, ages: np.ndarray, output_path: str,
              n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'euclidean',
              bprp0_min: float = None, bprp0_max: float = None, bprp0: np.ndarray = None):
    """Create UMAP embedding and plot colored by stellar age.

    UMAP is computed on ALL samples with valid ages first, then BPRP0 filtering
    is applied for plotting. This preserves consistent UMAP coordinates across
    different BPRP0 cuts.
    """
    import umap  # Lazy import to avoid requiring umap for non-plotting scripts

    # Filter to only include samples with valid ages for UMAP computation
    valid_mask = ~np.isnan(ages)
    latent_valid = latent_vectors[valid_mask]
    ages_valid = ages[valid_mask]
    bprp0_valid = bprp0[valid_mask] if bprp0 is not None else None
    
    print(f'Computing UMAP on {len(latent_valid)} samples with valid ages...')
    
    # Fit UMAP on ALL samples (before BPRP0 filtering)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=42,
        verbose=True
    )
    embedding = reducer.fit_transform(latent_valid)
    
    print(f'UMAP embedding shape: {embedding.shape}')
    
    # Now apply BPRP0 filtering for plotting (after UMAP)
    plot_mask = np.ones(len(ages_valid), dtype=bool)
    if bprp0_valid is not None:
        if bprp0_min is not None:
            plot_mask &= (bprp0_valid >= bprp0_min)
        if bprp0_max is not None:
            plot_mask &= (bprp0_valid <= bprp0_max)
    
    embedding_plot = embedding[plot_mask]
    ages_plot = ages_valid[plot_mask]
    
    n_filtered = len(ages_valid) - len(ages_plot)
    if n_filtered > 0:
        print(f'Filtered to {len(ages_plot)} samples for plotting (removed {n_filtered} outside BPRP0 range)')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use log scale for age coloring (ages span several orders of magnitude)
    scatter = ax.scatter(
        embedding_plot[:, 0],
        embedding_plot[:, 1],
        c=ages_plot,
        cmap='viridis',
        norm=LogNorm(vmin=ages_plot.min(), vmax=ages_plot.max()),
        s=50,
        alpha=0.7
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='Stellar Age (Myr)')
    cbar.ax.set_ylabel('Stellar Age (Myr)', fontsize=12)
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP of Light Curve Latent Space\nColored by Stellar Age', fontsize=14)
    
    # Add some statistics as text
    textstr = f'N samples: {len(ages_plot)}\nAge range: {ages_plot.min():.1f} - {ages_plot.max():.1f} Myr'
    if bprp0_min is not None or bprp0_max is not None:
        bprp0_str = f'BPRP0: {bprp0_min if bprp0_min is not None else "—"} to {bprp0_max if bprp0_max is not None else "—"}'
        textstr += f'\n{bprp0_str}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved UMAP plot to {output_path}')
    
    # Save the FULL embedding data (before BPRP0 filtering) for further analysis
    # This allows replotting with different BPRP0 cuts without recomputing UMAP
    npz_path = output_path.replace('.png', '_data.npz')
    np.savez(npz_path, 
             embedding=embedding,  # Full embedding (all samples)
             ages=ages_valid,      # All ages
             bprp0=bprp0_valid if bprp0_valid is not None else np.array([]),  # All BPRP0
             latent_vectors=latent_valid)
    print(f'Saved embedding data to {npz_path}')
    
    return embedding, ages_valid


def main():
    parser = argparse.ArgumentParser(description='UMAP visualization of latent space colored by stellar age')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--h5_path', type=str, default='data/timeseries.h5', help='Path to H5 data file')
    parser.add_argument('--pickle_path', type=str, default='data/star_sector_lc_formatted.pickle', help='Path to pickle file with metadata')
    parser.add_argument('--age_csv_path', type=str, default='data/TIC_cf_data.csv', help='Path to CSV with stellar ages')
    parser.add_argument('--output_dir', type=str, default='output/simple_rnn/umap_plots', help='Output directory')
    parser.add_argument('--hidden_size', type=int, default=64, help='Model hidden size')
    parser.add_argument('--direction', type=str, default='bi', choices=['forward', 'backward', 'bi'], help='Model direction')
    parser.add_argument('--mode', type=str, default='parallel', choices=['sequential', 'parallel'], help='Model mode')
    parser.add_argument('--use_flow', action='store_true', help='Whether model uses flow head')
    parser.add_argument('--max_length', type=int, default=2048, help='Max sequence length for processing')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--n_neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP min_dist parameter')
    parser.add_argument('--pooling_mode', type=str, default='multiscale', 
                        choices=['mean', 'multiscale', 'final'],
                        help='How to aggregate hidden states: mean (original), multiscale (recommended), final')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for faster testing). None = use all.')
    parser.add_argument('--stratify_by_age', action='store_true',
                        help='If set, sample evenly across log-age bins instead of taking first N samples')
    parser.add_argument('--n_age_bins', type=int, default=20,
                        help='Number of age bins for stratified sampling')
    parser.add_argument('--bprp0_min', type=float, default=None,
                        help='Minimum BPRP0 value (BP-RP color). None = no lower cut.')
    parser.add_argument('--bprp0_max', type=float, default=None,
                        help='Maximum BPRP0 value (BP-RP color). None = no upper cut.')
    parser.add_argument('--use_metadata', action='store_true',
                        help='Use stellar metadata (G_0, BP_0, RP_0, parallax + uncertainties) as model input')
    parser.add_argument('--use_conv_channels', action='store_true',
                        help='Use convolutional encoders for power spectrum, f-statistic, and ACF as additional model input')
    parser.add_argument('--conv_encoder_type', type=str, default='unet', choices=['lightweight', 'unet'],
                        help="Conv encoder architecture: 'unet' (default, richer but slower) or 'lightweight' (faster but simpler)")

    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which samples to use for UMAP (no BPRP0 filtering here)
    # BPRP0 filtering is applied AFTER UMAP to preserve consistent embedding
    sample_indices = None
    if args.max_samples is not None:
        # Load all ages first for stratified sampling (ignore BPRP0 at this stage)
        print('Loading ages for stratified sampling...')
        all_ages, all_bprp0, _, _ = load_ages(args.pickle_path, args.age_csv_path, sample_indices=None)
        
        if args.stratify_by_age:
            # Stratified sampling based on age only (not BPRP0)
            sample_indices = get_stratified_indices(
                all_ages, 
                max_samples=args.max_samples,
                n_bins=args.n_age_bins
            )
        else:
            # Simple sequential sampling from samples with valid ages
            valid_indices = np.where(~np.isnan(all_ages))[0]
            sample_indices = valid_indices[:args.max_samples]
    
    # Filter sample_indices to only include valid HDF5 indices
    if sample_indices is not None:
        with h5py.File(args.h5_path, 'r') as f:
            h5_size = len(f['flux'])
        valid_mask = sample_indices < h5_size
        if not np.all(valid_mask):
            n_invalid = np.sum(~valid_mask)
            print(f'Warning: {n_invalid} sample indices exceed HDF5 size ({h5_size}). Filtering to valid indices.')
            sample_indices = sample_indices[valid_mask]

    # Load model (num_meta_features will be overridden by checkpoint if stored there)
    num_meta_features = len(DEFAULT_METADATA_FIELDS) if args.use_metadata else 0

    # Build conv_config if using convolutional channels
    conv_config = None
    if args.use_conv_channels:
        if args.conv_encoder_type == 'lightweight':
            conv_config = {
                'encoder_type': 'lightweight',
                'hidden_channels': 16,
                'num_layers': 3,
                'activation': 'gelu'
            }
        else:  # unet
            conv_config = {
                'encoder_type': 'unet',
                'input_length': 16000,
                'encoder_dims': [4, 8, 16, 32],
                'num_layers': 4,
                'activation': 'gelu'
            }

    model = load_model(
        args.model_path,
        args.hidden_size,
        args.direction,
        args.mode,
        args.use_flow,
        device,
        num_meta_features=num_meta_features,
        use_conv_channels=args.use_conv_channels,
        conv_config=conv_config
    )
    
    # Extract latent vectors (for all selected samples, no BPRP0 filtering)
    latent_vectors = extract_latent_vectors(
        model,
        args.h5_path,
        device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        pooling_mode=args.pooling_mode,
        sample_indices=sample_indices,
        use_metadata=args.use_metadata,
        use_conv_channels=args.use_conv_channels
    )
    
    # Load ages and BPRP0 for selected samples
    ages, bprp0, tic_ids, prot = load_ages(args.pickle_path, args.age_csv_path, sample_indices=sample_indices)
    
    # Generate output filename based on model name, pooling mode, and cuts
    model_name = Path(args.model_path).stem
    suffix = f'{args.pooling_mode}'
    if args.bprp0_min is not None or args.bprp0_max is not None:
        bprp_range = f'bprp{args.bprp0_min or ""}-{args.bprp0_max or ""}'
        suffix += f'_{bprp_range}'
    output_path = output_dir / f'umap_age_{model_name}_{suffix}.png'
    
    # Create UMAP plot (BPRP0 filtering happens inside plot_umap after UMAP is computed)
    plot_umap(
        latent_vectors, 
        ages, 
        str(output_path),
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        bprp0_min=args.bprp0_min,
        bprp0_max=args.bprp0_max,
        bprp0=bprp0  # Pass BPRP0 for post-UMAP filtering
    )
    
    print('Done!')


if __name__ == '__main__':
    main()
