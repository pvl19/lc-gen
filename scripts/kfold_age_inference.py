"""K-fold cross-validation for age prediction from latent space + BPRP0.

This script trains k separate models, each on (k-1)/k of the data, and generates
age predictions for the held-out fold. The result is an unbiased age prediction
for every star, made by a model that never saw that star during training.

Star aggregation modes (--star_aggregation):
  none             - One sample per LC/sector; kfold splits by sector (legacy, has leakage
                     for multi-sector stars). Use only for backward compatibility.
  predict_mean     - One sample per LC/sector; kfold splits by *star* (no leakage);
                     per-sector predictions are averaged per star before reporting metrics.
  latent_mean      - Per-sector latent vectors are mean-pooled per star before kfold.
  latent_median    - Per-sector latent vectors are median-pooled per star before kfold.
  cross_sector     - Hidden states from all sectors of a star are concatenated before
                     multiscale pooling, yielding one latent vector per star.

Usage:
    python scripts/kfold_age_inference.py \\
        --model_path output/performance_tests/cf_final/model.pt \\
        --output_dir output/age_predictor/kfold \\
        --star_aggregation cross_sector \\
        --n_folds 10
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

try:
    import zuko
except ImportError:
    zuko = None

# Ensure local 'src' is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

# ---------------------------------------------------------------------------
# Age grid and outlier model constants (mirrors ChronoFlow convention)
# ---------------------------------------------------------------------------
PRIOR_LOGA_MYR = (0.0, 4.14)          # log10(1 Myr) → log10(~14 Gyr)
LOGA_GRID_DEFAULT_SIZE = 1000
LOGA_GRID = np.linspace(PRIOR_LOGA_MYR[0], PRIOR_LOGA_MYR[1], LOGA_GRID_DEFAULT_SIZE)
P_CLUSTER_MEM = 0.9                    # assumed cluster membership probability
P_OUTLIER     = 0.05                   # outlier fraction per star


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AgePredictor(nn.Module):
    """Neural Likelihood Estimation: MLP compressor + NSF modelling p(z | log10_age, BPRP0).

    Architecture (Neural Likelihood Estimation, NLE):
      - MLP encoder: latent_dim → hidden_dims → hidden_dims[-1]  (compressed z)
      - NSF flow:    p(z | log10_age, bprp0_norm)  (2-D context)

    Training: flow is conditioned on the *true* (log10_age, bprp0_norm) and trained
    to reconstruct the compressed latent z.  An outlier mixture model with fixed
    weights (P_CLUSTER_MEM=0.9, P_OUTLIER=0.05) is applied following ChronoFlow.

    Inference: the likelihood p(z | age_g, bprp0) is evaluated on a dense age grid
    and multiplied by a uniform log-age prior to obtain a discrete posterior, from
    which median/mean/MAP/p16/p84 are extracted.
    """

    def __init__(self, latent_dim: int, hidden_dims: list = [128, 64, 8], dropout: float = 0.2,
                 flow_transforms: int = 4, flow_hidden_features: list = None):
        super().__init__()
        if zuko is None:
            raise ImportError("zuko is required for the flow head. Install with: pip install zuko")
        if flow_hidden_features is None:
            flow_hidden_features = [64, 64]

        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.z_dim = hidden_dims[-1]

        # NSF: features = compressed z dim, context = (log10_age, bprp0_norm, bprp0_err_norm)
        self.flow = zuko.flows.NSF(
            features=self.z_dim,
            context=3,
            transforms=flow_transforms,
            hidden_features=flow_hidden_features,
        )

    def _nll_with_outlier(self, log_prob_flow: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Outlier mixture loss following ChronoFlow convention.

        Mixture: nf_weight * p_flow(z) + (1 - nf_weight) * p_outlier(z)
        Outlier distribution: isotropic standard normal over z (integrates to 1).
        """
        nf_weight = P_CLUSTER_MEM * (1.0 - P_OUTLIER)   # 0.855, constant
        ln_p_outlier = (
            -0.5 * z.pow(2).sum(dim=-1)
            - 0.5 * self.z_dim * np.log(2.0 * np.pi)
        )
        ln_p_cond_weighted = nf_weight * log_prob_flow
        ln_p_out           = np.log(1.0 - nf_weight) + ln_p_outlier
        ln_p_combined = torch.logsumexp(
            torch.stack([ln_p_cond_weighted, ln_p_out], dim=0), dim=0
        )
        return -ln_p_combined.mean()

    def forward(self, latent: torch.Tensor, log_age: torch.Tensor,
                bprp0_norm: torch.Tensor, bprp0_err_norm: torch.Tensor) -> torch.Tensor:
        """Compute NLL training loss.

        Args:
            latent:        (B, latent_dim)  normalised latent vectors
            log_age:       (B,)             log10(age/Myr) — used as flow context
            bprp0_norm:    (B,)             normalised BPRP0 — used as flow context
            bprp0_err_norm:(B,)             normalised BPRP0_err — used as flow context
        Returns:
            scalar NLL loss (with outlier model)
        """
        z       = self.encoder(latent)                                          # (B, z_dim)
        context = torch.stack([log_age, bprp0_norm, bprp0_err_norm], dim=1)    # (B, 3)
        dist    = self.flow(context)
        return self._nll_with_outlier(dist.log_prob(z), z)

    def predict_stats(self, latent: torch.Tensor, bprp0_norm: torch.Tensor,
                      bprp0_err_norm: torch.Tensor, loga_grid: torch.Tensor) -> dict:
        """Grid-based posterior inference: p(age | z, bprp0) ∝ p(z | age, bprp0) · p(age).

        Evaluates the flow likelihood over the full age grid in a single batched
        forward pass, applies a uniform log-age prior (flat → cancels in normalisation),
        and extracts summary statistics from the discrete posterior.

        Args:
            latent:        (B, latent_dim)
            bprp0_norm:    (B,)
            bprp0_err_norm:(B,)
            loga_grid:     (G,) log10(age/Myr) evaluation grid
        Returns:
            dict of (B,) tensors: median, mean, map, p16, p84 in log10(age/Myr)
        """
        z = self.encoder(latent)   # (B, z_dim)
        B, D = z.shape
        G    = loga_grid.shape[0]

        # Expand to (B*G, ...) for a single batched flow evaluation
        z_exp         = z.unsqueeze(1).expand(B, G, D).reshape(B * G, D)
        bprp0_exp     = bprp0_norm.unsqueeze(1).expand(B, G).reshape(B * G)
        bprp0_err_exp = bprp0_err_norm.unsqueeze(1).expand(B, G).reshape(B * G)
        loga_exp      = loga_grid.unsqueeze(0).expand(B, G).reshape(B * G)
        context       = torch.stack([loga_exp, bprp0_exp, bprp0_err_exp], dim=1)  # (B*G, 3)

        log_lik = self.flow(context).log_prob(z_exp).reshape(B, G)  # (B, G)

        # Uniform prior in log-age is constant → cancels; normalise directly
        log_post  = log_lik - torch.logsumexp(log_lik, dim=1, keepdim=True)
        posterior = log_post.exp()                                   # (B, G)

        # MAP
        map_idx = posterior.argmax(dim=1)
        map_est = loga_grid[map_idx]

        # Mean
        mean = (posterior * loga_grid.unsqueeze(0)).sum(dim=1)

        # CDF-based percentiles
        cdf = posterior.cumsum(dim=1)

        def pct(p: float) -> torch.Tensor:
            idx = (cdf >= p).float().argmax(dim=1)
            return loga_grid[idx]

        return {
            'median': pct(0.5),
            'mean':   mean,
            'map':    map_est,
            'p16':    pct(0.16),
            'p84':    pct(0.84),
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_latents_by_star(model, h5_path: str, device: torch.device,
                             sample_indices: np.ndarray, tic_ids_sorted: np.ndarray,
                             max_length: int, pooling_mode: str,
                             batch_size: int = 32,
                             use_metadata: bool = False, use_conv_channels: bool = False):
    """Single-pass extraction of per-sector AND cross-sector latents.

    Stars are accumulated into forward-pass batches of ~batch_size sectors so
    the GPU stays well-utilised.  After each forward pass the hidden states are
    split back into per-star groups; per-sector latents are computed immediately
    and the hidden states for each star are concatenated to produce the
    cross-sector latent.  Hidden states are discarded as soon as both latent
    types for a star are produced — peak memory is O(batch_size sectors), not
    O(total sectors).

    Args:
        model:             trained BiDirectionalMinGRU (eval mode expected)
        h5_path:           path to H5 data file
        device:            torch device
        sample_indices:    H5 indices in sorted order (same ordering as tic_ids_sorted)
        tic_ids_sorted:    TIC IDs aligned to sample_indices
        max_length:        per-sector max sequence length
        pooling_mode:      'mean' | 'multiscale' | 'final' — applied per sector
        batch_size:        target number of sectors per forward pass (stars are kept
                           whole, so actual batch size may slightly exceed this)
        use_metadata:      pass stellar metadata to model if available
        use_conv_channels: pass conv channels to model if available

    Returns:
        per_sector_latents:   (N, D) — one row per sector, same order as sample_indices
        cross_sector_latents: (S, D) — one row per unique star
        unique_tic_ids:       (S,)   — TIC ordering for cross_sector_latents
    """
    from plot_umap_latent import compute_multiscale_features
    from lcgen.models.TimeSeriesDataset import METADATA_FEATURES

    load_metadata = use_metadata and (model.meta_encoder is not None)
    N = len(sample_indices)

    # Load all H5 data for selected samples up front (single sequential read)
    with h5py.File(h5_path, 'r') as f:
        flux_all     = f['flux'][sample_indices]
        flux_err_all = f['flux_err'][sample_indices]
        time_all     = f['time'][sample_indices]
        lengths      = f['length'][sample_indices]

        if use_conv_channels:
            power_all  = f['power'][sample_indices]
            acf_all    = f['acf'][sample_indices]
            f_stat_all = f['f_stat'][sample_indices]
        else:
            power_all = acf_all = f_stat_all = None

        if load_metadata and 'metadata' in f:
            meta_grp  = f['metadata']
            meta_list = []
            for feat in METADATA_FEATURES:
                if feat in meta_grp:
                    vals = meta_grp[feat][sample_indices]
                    if vals.dtype.kind == 'S':
                        vals = vals.astype(float)
                    meta_list.append(vals.astype(np.float32))
                else:
                    meta_list.append(np.zeros(N, dtype=np.float32))
            metadata_all = np.nan_to_num(np.stack(meta_list, axis=1), nan=0.0)
        else:
            metadata_all = None

    actual_max = min(max_length, flux_all.shape[1])

    # Group positions (0..N-1) by TIC; preserve unique_tics order for output alignment
    unique_tics = np.unique(tic_ids_sorted)
    tic_to_pos  = {t: np.where(tic_ids_sorted == t)[0] for t in unique_tics}

    per_sector_latents   = [None] * N
    cross_sector_latents = []

    def _run_buffer(star_buffer):
        """Forward pass for a list of (pos_array, tic) pairs; fills per_sector_latents
        and appends to cross_sector_latents in the same order as star_buffer."""
        all_pos    = np.concatenate([pos for pos, _ in star_buffer])
        star_sizes = [len(pos) for pos, _ in star_buffer]

        flux_b = torch.tensor(flux_all[all_pos, :actual_max],     dtype=torch.float32, device=device)
        ferr_b = torch.tensor(flux_err_all[all_pos, :actual_max], dtype=torch.float32, device=device)
        time_b = torch.tensor(time_all[all_pos, :actual_max],     dtype=torch.float32, device=device)
        lens_b = np.minimum(lengths[all_pos], actual_max)

        B, L = flux_b.shape
        mask = torch.zeros(B, L, device=device)
        for j, vlen in enumerate(lens_b):
            mask[j, :vlen] = 1.0

        x_in = torch.stack([flux_b, ferr_b], dim=-1)
        t_in = time_b.unsqueeze(-1)

        meta_batch = (torch.tensor(metadata_all[all_pos], dtype=torch.float32, device=device)
                      if metadata_all is not None else None)

        if use_conv_channels and power_all is not None:
            conv_data_batch = {
                'power':  torch.tensor(power_all[all_pos,  :actual_max], dtype=torch.float32, device=device),
                'acf':    torch.tensor(acf_all[all_pos,    :actual_max], dtype=torch.float32, device=device),
                'f_stat': torch.tensor(f_stat_all[all_pos, :actual_max], dtype=torch.float32, device=device),
            }
        else:
            conv_data_batch = None

        out   = model(x_in, t_in, mask=mask, metadata=meta_batch,
                      conv_data=conv_data_batch, return_states=True)
        h_fwd = out.get('h_fwd_tensor')   # (B, L, H) or None
        h_bwd = out.get('h_bwd_tensor')   # (B, L, H) or None

        # Split hidden states back into per-star chunks
        h_fwd_split = torch.split(h_fwd, star_sizes, dim=0) if h_fwd is not None else [None] * len(star_buffer)
        h_bwd_split = torch.split(h_bwd, star_sizes, dim=0) if h_bwd is not None else [None] * len(star_buffer)

        offset = 0
        for (pos, _), hf, hb in zip(star_buffer, h_fwd_split, h_bwd_split):
            h_sectors = []
            for j in range(len(pos)):
                vlen = int(lens_b[offset + j])
                if hf is not None and hb is not None:
                    h = torch.cat([hf[j, :vlen, :], hb[j, :vlen, :]], dim=-1)
                elif hf is not None:
                    h = hf[j, :vlen, :]
                else:
                    h = hb[j, :vlen, :]

                if pooling_mode == 'mean':
                    latent = h.mean(dim=0)
                elif pooling_mode == 'final':
                    latent = h[-1, :]
                elif pooling_mode == 'multiscale':
                    latent = compute_multiscale_features(h, n_segments=4)
                else:
                    raise ValueError(f'Unknown pooling_mode: {pooling_mode}')

                per_sector_latents[pos[j]] = latent.cpu().numpy()
                h_sectors.append(h)

            h_all = torch.cat(h_sectors, dim=0)
            cross_sector_latents.append(compute_multiscale_features(h_all, n_segments=4).cpu().numpy())
            offset += len(pos)
            # h_sectors / h_all go out of scope — hidden states freed

    model.eval()
    print(f'Extracting latents for {len(unique_tics)} stars '
          f'({N} sectors total, pooling={pooling_mode}, batch_size~{batch_size})...')

    star_buffer   = []
    buffered_secs = 0
    stars_done    = 0

    with torch.no_grad():
        for tic in unique_tics:
            pos = tic_to_pos[tic]
            star_buffer.append((pos, tic))
            buffered_secs += len(pos)

            if buffered_secs >= batch_size:
                _run_buffer(star_buffer)
                stars_done   += len(star_buffer)
                star_buffer   = []
                buffered_secs = 0
                if stars_done % 500 < batch_size:   # approx every 500 stars
                    print(f'  {stars_done}/{len(unique_tics)} stars processed')

        if star_buffer:   # flush remainder
            _run_buffer(star_buffer)

    per_sector_arr   = np.stack(per_sector_latents)
    cross_sector_arr = np.stack(cross_sector_latents)
    print(f'Done: {N} per-sector latents {per_sector_arr.shape}, '
          f'{len(unique_tics)} cross-sector latents {cross_sector_arr.shape}')
    return per_sector_arr, cross_sector_arr, unique_tics


def load_data_fresh(model_path: str, h5_path: str, csv_path: str,
                    hidden_size: int, direction: str, mode: str, use_flow: bool,
                    max_length: int, batch_size: int, pooling_mode: str,
                    max_samples: int = None, stratify_by_age: bool = False,
                    n_age_bins: int = 20, bprp0_min: float = None, bprp0_max: float = None,
                    use_metadata: bool = False, use_conv_channels: bool = False,
                    conv_config: dict = None, require_prot: bool = False,
                    device: torch.device = None):
    """Load model and extract per-sector and cross-sector latent vectors.

    Returns:
        latent_vectors:          (N, D) per-sector latent vectors
        cross_sector_latents:    (S, D) cross-sector latent vectors (one per unique star)
        cross_sector_tic_ids:    (S,)   TIC ordering for cross_sector_latents
        ages:                    (N,) stellar ages in Myr
        bprp0:                   (N,) BP-RP0 colour
        gaia_ids:                (N,) GaiaDR3_ID strings
        tic_ids:                 (N,) TESS TIC IDs (int64)
        sectors:                 (N,) TESS sector numbers (int64)
    """
    from plot_umap_latent import load_model, load_ages, get_stratified_indices
    from lcgen.models.TimeSeriesDataset import METADATA_FEATURES

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Step 1: load full metadata to build the valid-sample mask ----------
    all_ages, all_bprp0, all_bprp0_err, all_gaia_ids, _, _ = load_ages(h5_path, csv_path, sample_indices=None)

    valid_mask = ~np.isnan(all_ages) & ~np.isnan(all_bprp0) & ~np.isnan(all_bprp0_err)
    if bprp0_min is not None:
        valid_mask &= (all_bprp0 >= bprp0_min)
    if bprp0_max is not None:
        valid_mask &= (all_bprp0 <= bprp0_max)

    if require_prot:
        df_csv = pd.read_csv(csv_path)
        df_csv['GaiaDR3_ID'] = df_csv['GaiaDR3_ID'].astype(str)
        id_to_prot = dict(zip(df_csv['GaiaDR3_ID'], df_csv['Prot']))
        all_prot = np.array([id_to_prot.get(g, np.nan) for g in all_gaia_ids])
        valid_mask &= (~np.isnan(all_prot) & (all_prot > 0))
        print(f'Filtering to stars with rotation periods: {valid_mask.sum()} valid samples')

    # -- Step 2: determine sample indices -----------------------------------
    if max_samples is not None:
        if stratify_by_age:
            ages_for_strat = np.where(valid_mask, all_ages, np.nan)
            sample_indices = get_stratified_indices(ages_for_strat, max_samples, n_age_bins)
        else:
            sample_indices = np.where(valid_mask)[0][:max_samples]
    else:
        sample_indices = np.where(valid_mask)[0]

    # Guard against out-of-bounds indices
    with h5py.File(h5_path, 'r') as f:
        h5_size = len(f['flux'])
    ok = sample_indices < h5_size
    if not np.all(ok):
        print(f'Warning: {(~ok).sum()} indices exceed H5 size — dropped.')
        sample_indices = sample_indices[ok]

    # -- Step 3: load metadata for selected samples -------------------------
    # Done before extraction so tic_ids are available for star-grouped processing.
    ages, bprp0, bprp0_err, gaia_ids, tic_ids, sectors = load_ages(h5_path, csv_path, sample_indices)

    # Defensive validity filter (load_ages returns sorted order matching sorted sample_indices)
    final_ok        = ~np.isnan(ages) & ~np.isnan(bprp0) & ~np.isnan(bprp0_err)
    sorted_indices  = np.sort(sample_indices)[final_ok]
    ages      = ages[final_ok]
    bprp0     = bprp0[final_ok]
    bprp0_err = bprp0_err[final_ok]
    gaia_ids  = gaia_ids[final_ok]
    tic_ids   = tic_ids[final_ok]
    sectors   = sectors[final_ok]

    # -- Step 4: load model -------------------------------------------------
    num_meta_features = len(METADATA_FEATURES) if use_metadata else 0
    model = load_model(model_path, device,
                       hidden_size=hidden_size, direction=direction, mode=mode,
                       use_flow=use_flow, num_meta_features=num_meta_features,
                       use_conv_channels=use_conv_channels, conv_config=conv_config)

    actual_use_metadata = model.num_meta_features > 0
    actual_use_conv     = model.use_conv_channels

    # -- Step 5: single-pass extraction (per-sector + cross-sector) ---------
    print('\n=== Extracting latents (per-sector + cross-sector in one pass) ===')
    latent_vectors, cross_sector_latents, cross_sector_tic_ids = extract_latents_by_star(
        model, h5_path, device,
        sample_indices=sorted_indices,
        tic_ids_sorted=tic_ids,
        max_length=max_length,
        pooling_mode=pooling_mode,
        batch_size=batch_size,
        use_metadata=actual_use_metadata,
        use_conv_channels=actual_use_conv,
    )

    print(f'Loaded {len(ages)} LC samples | latent dim: {latent_vectors.shape[1]}')
    print(f'Age range: {ages.min():.1f} – {ages.max():.1f} Myr')
    print(f'Unique stars: {len(np.unique(tic_ids))}')

    return latent_vectors, cross_sector_latents, cross_sector_tic_ids, ages, bprp0, bprp0_err, gaia_ids, tic_ids, sectors


# ---------------------------------------------------------------------------
# Latents cache  (save once, load many times)
# ---------------------------------------------------------------------------

def save_latents_cache(cache_path: str,
                       latent_vectors, ages, bprp0, bprp0_err, gaia_ids, tic_ids, sectors,
                       cross_sector_latents=None, cross_sector_tic_ids=None):
    """Save per-sector latents + identifiers (and optionally cross-sector latents) to npz.

    The cache contains everything needed for all star_aggregation modes so that
    the model never needs to be re-run when comparing methods.

    Layout
    ------
    latent_vectors        (N, D)          per-sector pooled latents
    ages                  (N,)            stellar age in Myr
    bprp0                 (N,)            BP-RP0
    gaia_ids              (N,)  str       GaiaDR3_ID
    tic_ids               (N,)  int64     TESS TIC
    sectors               (N,)  int64     TESS sector number
    cross_sector_latents  (S, D)          one multiscale latent per unique star  [optional]
    cross_sector_tic_ids  (S,)  int64     TIC ordering for cross_sector_latents  [optional]
    """
    save_dict = dict(
        latent_vectors=latent_vectors,
        ages=ages, bprp0=bprp0,
        gaia_ids=gaia_ids.astype(str),
        tic_ids=tic_ids, sectors=sectors,
    )
    if cross_sector_latents is not None:
        save_dict['cross_sector_latents'] = cross_sector_latents
        save_dict['cross_sector_tic_ids'] = cross_sector_tic_ids
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **save_dict)
    keys = list(save_dict.keys())
    print(f'Saved latents cache → {cache_path}  (keys: {keys})')


def load_latents_cache(cache_path: str):
    """Load a latents cache written by save_latents_cache.

    Returns
    -------
    latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors,
    cross_sector_latents (or None), cross_sector_tic_ids (or None)
    """
    data = np.load(cache_path, allow_pickle=True)
    cross_sector_latents  = data['cross_sector_latents']  if 'cross_sector_latents' in data else None
    cross_sector_tic_ids  = data['cross_sector_tic_ids']  if 'cross_sector_tic_ids'  in data else None
    print(f'Loaded latents cache ← {cache_path}')
    print(f'  per-sector samples : {len(data["ages"])}')
    print(f'  latent dim         : {data["latent_vectors"].shape[1]}')
    if cross_sector_latents is not None:
        print(f'  cross-sector stars : {len(cross_sector_tic_ids)}')
    else:
        print(f'  cross-sector latents: not in cache')
    return (data['latent_vectors'], data['ages'], data['bprp0'],
            data['gaia_ids'], data['tic_ids'], data['sectors'],
            cross_sector_latents, cross_sector_tic_ids)


# ---------------------------------------------------------------------------
# Star-level aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_by_star(latent_vectors, ages, bprp0, bprp0_err, tic_ids, method: str):
    """Reduce per-sector arrays to one row per unique star.

    Args:
        method: one of 'latent_mean', 'latent_median', 'latent_max',
                'latent_mean_std'.
                'latent_mean_std' concatenates [mean, std] along the feature
                axis, doubling the latent dimension. For single-sector stars
                the std is all zeros — the MLP sees this as a valid input but
                it carries no sector-variability information. Be aware this may
                introduce a confound if the number of sectors observed correlates
                with stellar properties.

    Returns:
        star_latents:    (n_stars, D)  — D is 2× input dim for latent_mean_std
        star_ages:       (n_stars,)
        star_bprp0:      (n_stars,)
        star_bprp0_err:  (n_stars,)
        unique_tic_ids:  (n_stars,)
    """
    unique_tics  = np.unique(tic_ids)
    star_latents  = []
    star_ages     = []
    star_bprp0    = []
    star_bprp0_err = []

    for tic in unique_tics:
        mask = tic_ids == tic
        sectors_latents = latent_vectors[mask]  # (n_sectors, D)
        if method == 'latent_mean':
            star_latents.append(sectors_latents.mean(axis=0))
        elif method == 'latent_median':
            star_latents.append(np.median(sectors_latents, axis=0))
        elif method == 'latent_max':
            star_latents.append(sectors_latents.max(axis=0))
        elif method == 'latent_mean_std':
            mean = sectors_latents.mean(axis=0)
            std  = sectors_latents.std(axis=0)   # zero for single-sector stars
            star_latents.append(np.concatenate([mean, std]))
        else:
            raise ValueError(f'Unknown aggregation method: {method}')
        star_ages.append(ages[mask][0])
        star_bprp0.append(bprp0[mask][0])
        star_bprp0_err.append(bprp0_err[mask][0])

    star_latents = np.stack(star_latents)
    print(f'{method}: reduced to {len(unique_tics)} stars | latent dim: {star_latents.shape[1]}')
    return star_latents, np.array(star_ages), np.array(star_bprp0), np.array(star_bprp0_err), unique_tics


def extract_cross_sector_latents(model, h5_path: str, device: torch.device,
                                  sample_indices: np.ndarray, tic_ids_sorted: np.ndarray,
                                  max_length: int = 16384):
    """Pool hidden states across ALL sectors of each star via multiscale aggregation.

    Concatenates the valid hidden states from every sector of a star into a single
    sequence, then applies the same multiscale pooling used per-sector. This lets
    the model see the full temporal context of the star before collapsing to a
    single latent vector.

    Args:
        model:           trained BiDirectionalMinGRU
        h5_path:         path to H5 data file
        device:          torch device
        sample_indices:  H5 indices in sorted order (same ordering load_data_fresh uses)
        tic_ids_sorted:  TIC IDs aligned to sample_indices (same sorted order)
        max_length:      per-sector max sequence length

    Returns:
        star_latents:    (n_unique_stars, D)
        unique_tic_ids:  (n_unique_stars,)
    """
    from plot_umap_latent import compute_multiscale_features

    sorted_idx = np.sort(sample_indices)

    with h5py.File(h5_path, 'r') as f:
        flux_all     = f['flux'][sorted_idx]
        flux_err_all = f['flux_err'][sorted_idx]
        time_all     = f['time'][sorted_idx]
        lengths      = f['length'][sorted_idx]

    # Group positions in sorted arrays by TIC
    unique_tics      = np.unique(tic_ids_sorted)
    tic_to_positions = {t: np.where(tic_ids_sorted == t)[0] for t in unique_tics}

    star_latents = []
    model.eval()

    print(f'Cross-sector extraction for {len(unique_tics)} stars...')

    with torch.no_grad():
        for i, tic in enumerate(unique_tics):
            pos = tic_to_positions[tic]  # indices into sorted arrays

            flux_b = torch.tensor(flux_all[pos],     dtype=torch.float32, device=device)
            ferr_b = torch.tensor(flux_err_all[pos], dtype=torch.float32, device=device)
            time_b = torch.tensor(time_all[pos],     dtype=torch.float32, device=device)
            lens_b = lengths[pos]

            actual_max = min(max_length, flux_b.shape[1])
            flux_b = flux_b[:, :actual_max]
            ferr_b = ferr_b[:, :actual_max]
            time_b = time_b[:, :actual_max]
            lens_b = np.minimum(lens_b, actual_max)

            B, L = flux_b.shape
            mask = torch.zeros(B, L, device=device)
            for j, vlen in enumerate(lens_b):
                mask[j, :vlen] = 1.0

            x_in = torch.stack([flux_b, ferr_b], dim=-1)
            t_in = time_b.unsqueeze(-1)

            out   = model(x_in, t_in, mask=mask, return_states=True)
            h_fwd = out.get('h_fwd_tensor')
            h_bwd = out.get('h_bwd_tensor')

            # Collect valid hidden states from every sector of this star
            h_segments = []
            for j in range(B):
                vlen = lens_b[j]
                if h_fwd is not None and h_bwd is not None:
                    h = torch.cat([h_fwd[j, :vlen, :], h_bwd[j, :vlen, :]], dim=-1)
                elif h_fwd is not None:
                    h = h_fwd[j, :vlen, :]
                else:
                    h = h_bwd[j, :vlen, :]
                h_segments.append(h)

            h_all  = torch.cat(h_segments, dim=0)  # (total_valid_timesteps, hidden)
            latent = compute_multiscale_features(h_all, n_segments=4)
            star_latents.append(latent.cpu().numpy())

            if (i + 1) % 250 == 0:
                print(f'  {i + 1}/{len(unique_tics)} stars processed')

    star_latents = np.stack(star_latents)
    print(f'Cross-sector latents: {len(unique_tics)} stars | shape {star_latents.shape}')
    return star_latents, unique_tics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_single_fold(X_train, y_train, bprp0_train, bprp0_err_train,
                      X_val, bprp0_val, bprp0_err_val,
                      hidden_dims, dropout, lr, weight_decay, n_epochs, batch_size, device,
                      flow_transforms=4, flow_hidden_features=None, loga_grid=None):
    """Train one fold NLE model, return validation posterior stats and best model state.

    X_train/X_val:            (N, latent_dim) normalised latent vectors
    y_train:                  (N,) log10(age/Myr) — used as flow context at training time
    bprp0_train/val:          (N,) normalised BPRP0 — used as flow context
    bprp0_err_train/val:      (N,) normalised BPRP0_err — used as flow context
    loga_grid:                (G,) torch.Tensor age grid for posterior inference
    """
    if loga_grid is None:
        loga_grid = torch.tensor(LOGA_GRID, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(
        torch.tensor(X_train,         dtype=torch.float32, device=device),
        torch.tensor(y_train,         dtype=torch.float32, device=device),
        torch.tensor(bprp0_train,     dtype=torch.float32, device=device),
        torch.tensor(bprp0_err_train, dtype=torch.float32, device=device),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = AgePredictor(X_train.shape[1], hidden_dims, dropout,
                         flow_transforms=flow_transforms,
                         flow_hidden_features=flow_hidden_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_loss  = float('inf')
    best_state = None

    for _ in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b, bprp0_b, bprp0_err_b in train_loader:
            optimizer.zero_grad()
            loss = model(X_b, y_b, bprp0_b, bprp0_err_b)   # NLE: p(z | age, bprp0, bprp0_err)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
        epoch_loss /= len(X_train)
        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        X_val_t         = torch.tensor(X_val,         dtype=torch.float32, device=device)
        bprp0_val_t     = torch.tensor(bprp0_val,     dtype=torch.float32, device=device)
        bprp0_err_val_t = torch.tensor(bprp0_err_val, dtype=torch.float32, device=device)
        loga_grid_d     = loga_grid.to(device)
        stats     = model.predict_stats(X_val_t, bprp0_val_t, bprp0_err_val_t, loga_grid_d)
        val_stats = {k: v.cpu().numpy() for k, v in stats.items()}

    return val_stats, best_loss, best_state


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------

def run_kfold_cv(latent_vectors, ages, bprp0, bprp0_err, tic_ids,
                 n_folds=10, hidden_dims=[128, 64, 8], dropout=0.2,
                 lr=1e-3, weight_decay=1e-4, n_epochs=100, batch_size=64,
                 device=None, seed=42, save_models_dir=None,
                 star_level_split=False,
                 flow_transforms=4, flow_hidden_features=None, loga_grid_size=None):
    """Run k-fold cross-validation.

    Args:
        star_level_split: if True, folds are assigned by unique TIC ID so all
                          sectors of a star always land in the same fold. This
                          avoids data leakage when the same star has multiple sectors.
                          Set to True for 'predict_mean'; irrelevant for aggregated
                          methods where there is already one row per star.

    Returns:
        predictions, true_ages, tic_ids, fold_assignments, fold_losses, norm_params
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(seed)
    n_samples = len(ages)

    # Normalise latent vectors (X).  BPRP0 is kept separate — it goes to the
    # flow as context alongside log10_age, not into the MLP encoder.
    X = latent_vectors
    y = np.log10(ages)

    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # bprp0: passed raw (~0–4 scale, matches log10_age range)
    # bprp0_err: log10 transform only (raw values are small; log10 brings to ~-2–0 scale)
    bprp0_norm     = bprp0
    bprp0_err_norm = np.log10(np.clip(bprp0_err, 1e-10, None))

    norm_params = {
        'X_mean': X_mean, 'X_std': X_std,
        'latent_dim': latent_vectors.shape[1],
        'input_dim': X.shape[1],
    }

    # Age grid for posterior inference
    G = loga_grid_size or LOGA_GRID_DEFAULT_SIZE
    loga_grid_np = np.linspace(PRIOR_LOGA_MYR[0], PRIOR_LOGA_MYR[1], G)
    loga_grid_t  = torch.tensor(loga_grid_np, dtype=torch.float32)

    # Build fold assignments
    fold_of_sample = np.zeros(n_samples, dtype=int)

    if star_level_split:
        unique_tics   = np.unique(tic_ids)
        perm          = np.random.permutation(len(unique_tics))
        tics_shuffled = unique_tics[perm]
        tic_fold_size = len(tics_shuffled) // n_folds
        tic_to_fold   = {}
        for f in range(n_folds):
            s = f * tic_fold_size
            e = s + tic_fold_size if f < n_folds - 1 else len(tics_shuffled)
            for t in tics_shuffled[s:e]:
                tic_to_fold[t] = f
        fold_of_sample = np.array([tic_to_fold[t] for t in tic_ids])
        print(f'Star-level split: {len(unique_tics)} unique stars across {n_folds} folds')
    else:
        perm      = np.random.permutation(n_samples)
        fold_size = n_samples // n_folds
        for f in range(n_folds):
            s = f * fold_size
            e = s + fold_size if f < n_folds - 1 else n_samples
            fold_of_sample[perm[s:e]] = f

    # One array per posterior statistic (all in log10_age space)
    stat_keys = ('median', 'mean', 'map', 'p16', 'p84')
    all_stats   = {k: np.zeros(n_samples) for k in stat_keys}
    fold_losses = []
    fold_models = []

    print(f'\nRunning {n_folds}-fold CV on {n_samples} samples '
          f'({"star-level" if star_level_split else "sample-level"} split)...')

    for fold in range(n_folds):
        val_idx   = np.where(fold_of_sample == fold)[0]
        train_idx = np.where(fold_of_sample != fold)[0]

        X_train, y_train   = X_norm[train_idx],        y[train_idx]
        X_val              = X_norm[val_idx]
        bprp0_train        = bprp0_norm[train_idx]
        bprp0_val          = bprp0_norm[val_idx]
        bprp0_err_train    = bprp0_err_norm[train_idx]
        bprp0_err_val      = bprp0_err_norm[val_idx]

        val_stats, train_loss, model_state = train_single_fold(
            X_train, y_train, bprp0_train, bprp0_err_train,
            X_val, bprp0_val, bprp0_err_val,
            hidden_dims, dropout, lr, weight_decay, n_epochs, batch_size, device,
            flow_transforms=flow_transforms, flow_hidden_features=flow_hidden_features,
            loga_grid=loga_grid_t,
        )
        fold_models.append(model_state)

        for k in stat_keys:
            all_stats[k][val_idx] = val_stats[k]
        fold_of_sample[val_idx] = fold  # already set, but explicit
        fold_losses.append(train_loss)

        val_pred_log = val_stats['median']
        val_mae  = np.mean(np.abs(val_pred_log - y[val_idx]))  # dex
        val_corr = np.corrcoef(val_pred_log, y[val_idx])[0, 1]
        print(f'  Fold {fold + 1}/{n_folds}: train_loss={train_loss:.4f}  '
              f'val_MAE={val_mae:.3f} dex  val_r={val_corr:.3f}')

    if save_models_dir is not None:
        save_dir = Path(save_models_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'fold_models': fold_models,
            'normalization_params': norm_params,
            'hidden_dims': hidden_dims,
            'dropout': dropout,
            'n_folds': n_folds,
            'seed': seed,
        }, save_dir / 'kfold_models.pt')
        print(f'Saved {n_folds} fold models to {save_dir / "kfold_models.pt"}')

    all_predictions = all_stats['median']  # log10(age/Myr), used as primary prediction
    return all_predictions, all_stats, ages, tic_ids, fold_of_sample, fold_losses, norm_params


# ---------------------------------------------------------------------------
# Results / plotting
# ---------------------------------------------------------------------------

def plot_kfold_results(predictions, true_ages, bprp0, tic_ids, fold_assignments, output_dir,
                       pred_stats=None):
    """Create diagnostic plots and save metrics + predictions CSV.

    predictions: log10(age/Myr) — median from flow posterior
    true_ages:   linear Myr (raw data)
    pred_stats:  dict of log10(age/Myr) arrays: median, mean, map, p16, p84
    """
    log_true  = np.log10(true_ages)
    log_resid = predictions - log_true   # dex (pred − true)
    mae       = np.mean(np.abs(log_resid))
    rmse      = np.sqrt(np.mean(log_resid ** 2))
    med_ae    = np.median(np.abs(log_resid))
    bias      = np.mean(log_resid)
    scat      = np.std(log_resid)
    corr      = np.corrcoef(predictions, log_true)[0, 1]

    print(f'\n=== K-Fold Results ===')
    print(f'N = {len(true_ages)}')
    print(f'MAE = {mae:.3f} dex   Median AE = {med_ae:.3f} dex   RMSE = {rmse:.3f} dex')
    print(f'Bias = {bias:.3f} dex   Scatter = {scat:.3f} dex   Pearson r = {corr:.3f}')

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    sc  = ax1.scatter(log_true, predictions, c=bprp0, cmap='coolwarm',
                      alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    lo, hi = log_true.min(), log_true.max()
    ax1.plot([lo, hi], [lo, hi], 'k--', lw=2, label='Perfect')
    ax1.plot([lo, hi], [lo + 0.3, hi + 0.3], 'k:', alpha=0.5, label='±0.3 dex')
    ax1.plot([lo, hi], [lo - 0.3, hi - 0.3], 'k:', alpha=0.5)
    ax1.set_xlabel('log₁₀(True Age / Myr)'); ax1.set_ylabel('log₁₀(Predicted Age / Myr)')
    ax1.set_title(f'K-Fold Age Predictions (r={corr:.3f})')
    ax1.legend(loc='upper left')
    plt.colorbar(sc, ax=ax1, label='BP-RP₀')
    ax1.text(0.02, 0.98,
             f'N={len(true_ages)}\nMAE={mae:.3f} dex\nσ={scat:.3f} dex\nr={corr:.3f}',
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(log_true, log_resid, c=bprp0, cmap='coolwarm',
                alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    ax2.axhline(0, color='k', linestyle='--', lw=2)
    ax2.set_xlabel('log₁₀(True Age / Myr)'); ax2.set_ylabel('Residual (dex)')
    ax2.set_title('Residuals vs True Age')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(log_resid, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='r', linestyle='--', lw=2, label='Zero')
    ax3.axvline(bias, color='orange', lw=2, label=f'Bias={bias:.3f}')
    ax3.set_xlabel('Residual (dex)'); ax3.set_ylabel('Count')
    ax3.set_title(f'Residuals (σ={scat:.3f} dex)')
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(bprp0, log_resid, c=log_true, cmap='viridis', alpha=0.6, s=15)
    ax4.axhline(0, color='k', linestyle='--', lw=2)
    ax4.set_xlabel('BP-RP₀'); ax4.set_ylabel('Residual (dex)')
    ax4.set_title('Residuals vs Colour (by log age)')
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(log_true.min(), log_true.max()))
    plt.colorbar(sm, ax=ax4, label='log₁₀(Age / Myr)')

    plt.tight_layout()
    plt.savefig(output_dir / 'kfold_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {output_dir / "kfold_results.png"}')

    df = pd.DataFrame({
        'TIC_ID':             tic_ids,
        'log10_true_age':     log_true,
        'log10_pred_median':  predictions,
        'residual_dex':       log_resid,
        'bprp0':              bprp0,
        'fold':               fold_assignments,
    })
    if pred_stats is not None:
        df['log10_pred_mean'] = pred_stats['mean']
        df['log10_pred_map']  = pred_stats['map']
        df['log10_pred_p16']  = pred_stats['p16']
        df['log10_pred_p84']  = pred_stats['p84']
    df.to_csv(output_dir / 'kfold_predictions.csv', index=False)
    print(f'Saved predictions to {output_dir / "kfold_predictions.csv"}')

    metrics = {
        'n_samples':     int(len(true_ages)),
        'mae_dex':       float(mae),
        'median_ae_dex': float(med_ae),
        'rmse_dex':      float(rmse),
        'bias_dex':      float(bias),
        'scatter_dex':   float(scat),
        'correlation':   float(corr),
    }
    with open(output_dir / 'kfold_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='K-fold cross-validation for age prediction')

    # Data / model
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint. Required unless --load_latents is set.')
    parser.add_argument('--h5_path',    type=str, default='data/timeseries_x.h5')
    parser.add_argument('--age_csv',    type=str, default='data/phot_all.csv')
    parser.add_argument('--output_dir', type=str, default='output/age_predictor/kfold')

    # (kept for backward compat with old shell scripts; not used)
    parser.add_argument('--pickle_path', type=str, default=None,
                        help='Deprecated — ignored. Data is now loaded from --h5_path.')

    # Model extraction
    parser.add_argument('--hidden_size',      type=int,   default=64)
    parser.add_argument('--direction',        type=str,   default='bi',
                        choices=['forward', 'backward', 'bi'])
    parser.add_argument('--mode',             type=str,   default='parallel',
                        choices=['sequential', 'parallel'])
    parser.add_argument('--use_flow',         action='store_true')
    parser.add_argument('--use_metadata',     action='store_true')
    parser.add_argument('--use_conv_channels',action='store_true')
    parser.add_argument('--conv_encoder_type',type=str,   default='unet',
                        choices=['lightweight', 'unet'])
    parser.add_argument('--max_length',       type=int,   default=16384)
    parser.add_argument('--pooling_mode',     type=str,   default='multiscale',
                        choices=['mean', 'multiscale', 'final'])

    # Star aggregation
    parser.add_argument('--star_aggregation', type=str,   default='none',
                        choices=['none', 'predict_mean', 'latent_mean', 'latent_median',
                                 'latent_max', 'latent_mean_std', 'cross_sector'],
                        help=(
                            'none          – one sample per sector; kfold splits by sector '
                            '(legacy, has leakage for multi-sector stars). '
                            'predict_mean  – kfold splits by star; per-sector predictions '
                            'are averaged per star before metrics. '
                            'latent_mean   – sector latents are mean-pooled per star. '
                            'latent_median – sector latents are median-pooled per star. '
                            'cross_sector  – hidden states concatenated across sectors '
                            'before multiscale pooling.'
                        ))

    # Data selection
    parser.add_argument('--max_samples',     type=int,   default=None)
    parser.add_argument('--stratify_by_age', action='store_true')
    parser.add_argument('--n_age_bins',      type=int,   default=20)
    parser.add_argument('--bprp0_min',       type=float, default=None)
    parser.add_argument('--bprp0_max',       type=float, default=None)
    parser.add_argument('--require_prot',    action='store_true')

    # K-fold
    parser.add_argument('--n_folds',    type=int,   default=10)
    parser.add_argument('--seed',       type=int,   default=42)

    # Age predictor training
    parser.add_argument('--hidden_dims',          type=int,   nargs='+', default=[128, 64, 32])
    parser.add_argument('--dropout',              type=float, default=0.2)
    parser.add_argument('--lr',                   type=float, default=1e-3)
    parser.add_argument('--weight_decay',         type=float, default=1e-4)
    parser.add_argument('--n_epochs',             type=int,   default=100)
    parser.add_argument('--batch_size',           type=int,   default=64)
    parser.add_argument('--flow_transforms',      type=int,   default=4,
                        help='Number of NSF transforms in the flow head (default: 4)')
    parser.add_argument('--flow_hidden_dims',     type=int,   nargs='+', default=[64, 64],
                        help='Hidden layer sizes inside each flow transform (default: 64 64)')
    parser.add_argument('--loga_grid_size',       type=int,   default=LOGA_GRID_DEFAULT_SIZE,
                        help='Number of age grid points for posterior inference (default: 1000)')

    # Latents cache
    parser.add_argument('--save_latents', type=str, default=None, metavar='PATH',
                        help='After extracting latents, save them (+ cross-sector latents) '
                             'to this npz file. Pass to --load_latents on subsequent runs '
                             'to skip model inference entirely.')
    parser.add_argument('--load_latents', type=str, default=None, metavar='PATH',
                        help='Load pre-computed latents from this npz file instead of '
                             'running the model. --model_path is not required when this '
                             'flag is set.')

    args = parser.parse_args()

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device: {device}')

    conv_config = None
    if args.use_conv_channels:
        if args.conv_encoder_type == 'lightweight':
            conv_config = {'encoder_type': 'lightweight', 'hidden_channels': 16,
                           'num_layers': 3, 'activation': 'gelu'}
        else:
            conv_config = {'encoder_type': 'unet', 'input_length': 16000,
                           'encoder_dims': [4, 8, 16, 32], 'num_layers': 4,
                           'activation': 'gelu'}

    # ── Step 1: latents (from model or from cache) ──────────────────────────
    cached_cross_sector_latents = None
    cached_cross_sector_tic_ids = None

    if args.load_latents:
        print(f'\n=== Loading latents from cache: {args.load_latents} ===')
        (latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors,
         cached_cross_sector_latents, cached_cross_sector_tic_ids) = load_latents_cache(args.load_latents)
        # bprp0_err is not stored in the cache — load from CSV using gaia_ids
        _df = pd.read_csv(args.age_csv)
        _df['GaiaDR3_ID'] = _df['GaiaDR3_ID'].astype(str)
        _id_to_bprp0_err = dict(zip(_df['GaiaDR3_ID'], _df['BPRP0_err']))
        bprp0_err = np.array([_id_to_bprp0_err.get(g, np.nan) for g in gaia_ids], dtype=float)
    else:
        if args.model_path is None:
            parser.error('--model_path is required unless --load_latents is set.')
        (latent_vectors, cached_cross_sector_latents, cached_cross_sector_tic_ids,
         ages, bprp0, bprp0_err, gaia_ids, tic_ids, sectors) = load_data_fresh(
            model_path=args.model_path,
            h5_path=args.h5_path,
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
            use_metadata=args.use_metadata,
            use_conv_channels=args.use_conv_channels,
            conv_config=conv_config,
            require_prot=args.require_prot,
            device=device,
        )

        if args.save_latents:
            save_latents_cache(
                args.save_latents,
                latent_vectors, ages, bprp0, bprp0_err, gaia_ids, tic_ids, sectors,
                cached_cross_sector_latents, cached_cross_sector_tic_ids,
            )

    # ── Step 2: apply star aggregation ─────────────────────────────────────
    agg = args.star_aggregation
    star_level_split = False  # whether run_kfold_cv should split by unique TIC

    if agg == 'none':
        # Legacy: per-sector, kfold by sample (may have leakage)
        kfold_latents   = latent_vectors
        kfold_ages      = ages
        kfold_bprp0     = bprp0
        kfold_bprp0_err = bprp0_err
        kfold_tics      = tic_ids

    elif agg == 'predict_mean':
        # Per-sector latents + star-level kfold split; average preds per star after
        kfold_latents   = latent_vectors
        kfold_ages      = ages
        kfold_bprp0     = bprp0
        kfold_bprp0_err = bprp0_err
        kfold_tics      = tic_ids
        star_level_split = True

    elif agg in ('latent_mean', 'latent_median', 'latent_max', 'latent_mean_std'):
        print(f'\n=== Aggregating latents ({agg}) ===')
        kfold_latents, kfold_ages, kfold_bprp0, kfold_bprp0_err, kfold_tics = aggregate_by_star(
            latent_vectors, ages, bprp0, bprp0_err, tic_ids, method=agg
        )

    elif agg == 'cross_sector':
        print('\n=== Cross-sector hidden-state pooling ===')
        kfold_latents = cached_cross_sector_latents
        kfold_tics    = cached_cross_sector_tic_ids
        tic_to_age       = {t: ages[tic_ids == t][0]      for t in np.unique(tic_ids)}
        tic_to_bprp0     = {t: bprp0[tic_ids == t][0]     for t in np.unique(tic_ids)}
        tic_to_bprp0_err = {t: bprp0_err[tic_ids == t][0] for t in np.unique(tic_ids)}
        kfold_ages      = np.array([tic_to_age[t]       for t in kfold_tics])
        kfold_bprp0     = np.array([tic_to_bprp0[t]     for t in kfold_tics])
        kfold_bprp0_err = np.array([tic_to_bprp0_err[t] for t in kfold_tics])

    else:
        raise ValueError(f'Unknown star_aggregation: {agg}')

    # ── Step 3: k-fold cross-validation ────────────────────────────────────
    print('\n=== Running K-Fold Cross-Validation ===')
    predictions, pred_stats, true_ages, result_tics, fold_assignments, fold_losses, _ = run_kfold_cv(
        latent_vectors=kfold_latents,
        ages=kfold_ages,
        bprp0=kfold_bprp0,
        bprp0_err=kfold_bprp0_err,
        tic_ids=kfold_tics,
        n_folds=args.n_folds,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        save_models_dir=output_dir,
        star_level_split=star_level_split,
        flow_transforms=args.flow_transforms,
        flow_hidden_features=args.flow_hidden_dims,
        loga_grid_size=args.loga_grid_size,
    )

    # ── Step 4: for predict_mean, save per-sector CSV then average per star ─
    if agg == 'predict_mean':
        # Save per-sector predictions first — useful for uncertainty estimation
        # (spread of per-sector predictions = observational scatter per star)
        pd.DataFrame({
            'TIC_ID': result_tics,
            'sector': sectors,
            'true_age_myr': true_ages,
            'predicted_age_myr': predictions,
            'bprp0': kfold_bprp0,
            'fold': fold_assignments,
        }).to_csv(output_dir / 'kfold_predictions_per_sector.csv', index=False)
        print(f'Saved per-sector predictions to {output_dir / "kfold_predictions_per_sector.csv"}')

        print('\nAveraging per-sector predictions per star...')
        unique_tics = np.unique(result_tics)
        predictions     = np.array([predictions[result_tics == t].mean()       for t in unique_tics])
        true_ages       = np.array([true_ages[result_tics == t][0]             for t in unique_tics])
        kfold_bprp0     = np.array([kfold_bprp0[result_tics == t][0]           for t in unique_tics])
        kfold_bprp0_err = np.array([kfold_bprp0_err[result_tics == t][0]       for t in unique_tics])
        fold_assignments = np.array([fold_assignments[result_tics == t][0]     for t in unique_tics])
        result_tics     = unique_tics
        print(f'Averaged to {len(unique_tics)} unique stars')

    # ── Step 5: results ─────────────────────────────────────────────────────
    plot_kfold_results(
        predictions, true_ages, kfold_bprp0, result_tics, fold_assignments, output_dir,
        pred_stats=pred_stats,
    )

    config = vars(args)
    config.update({
        'latent_dim':    int(kfold_latents.shape[1]),
        'n_samples':     int(len(kfold_ages)),
        'fold_losses':   [float(l) for l in fold_losses],
        'star_level_split': star_level_split,
    })
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print('\nDone!')


if __name__ == '__main__':
    main()
