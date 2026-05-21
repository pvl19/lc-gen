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
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Ensure local 'src' is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lcgen.models.simple_min_gru import BiDirectionalMinGRU
from lcgen.models.TimeSeriesDataset import METADATA_FEATURES
from lcgen.models.MetadataAgePredictor import MetadataStandardizer


def save_latents_cache(cache_path: str, latent_vectors, ages, bprp0,
                       gaia_ids, tic_ids, sectors,
                       bprp0_err=None, mg=None, mg_err=None, mem_prob=None):
    """Save latents + identifiers to npz in the canonical kfold format.

    The resulting file is a drop-in cache for kfold_age_inference.py's
    ``--load_latents``. Optional fields (bprp0_err, mg, mg_err, mem_prob) are
    written when supplied; their absence triggers kfold's CSV-fill fallback.
    """
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    save_dict = dict(
        latent_vectors=np.asarray(latent_vectors),
        ages=np.asarray(ages),
        bprp0=np.asarray(bprp0),
        gaia_ids=np.asarray(gaia_ids).astype(str),
        tic_ids=np.asarray(tic_ids),
        sectors=np.asarray(sectors),
    )
    if bprp0_err is not None:
        save_dict['bprp0_err'] = np.asarray(bprp0_err)
    if mg is not None:
        save_dict['mg'] = np.asarray(mg)
    if mg_err is not None:
        save_dict['mg_err'] = np.asarray(mg_err)
    if mem_prob is not None:
        save_dict['mem_prob'] = np.asarray(mem_prob)
    np.savez(cache_path, **save_dict)
    print(f'Saved latents cache → {cache_path} ({len(ages)} samples, '
          f'keys={list(save_dict.keys())})')


def load_latents_cache(cache_path: str, age_csv_paths=None, return_extras: bool = False):
    """Load a latents cache. Accepts two schemas:

    - kfold/UMAP format: latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors
      (+ optional bprp0_err, mg, mem_prob)
    - predict_ages.py format: latents, gaia_ids, bprp0, [bprp0_err]

    Missing fields (ages, tic_ids, sectors, bprp0_err, mg) are filled from
    age_csv_paths via Gaia ID join when available; tic_ids/sectors fall back to
    -1 placeholders (UMAP plotting tolerates this — they're only used for hover
    labels in saved CSVs).

    Returns (latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors). When
    return_extras=True, additionally returns (bprp0_err, mg, mem_prob) so the
    cache can be re-saved in canonical kfold format without information loss.
    """
    d = np.load(cache_path, allow_pickle=True)
    keys = set(d.keys())

    lv_key = 'latent_vectors' if 'latent_vectors' in keys else 'latents'
    latent_vectors = d[lv_key]
    gaia_ids = d['gaia_ids'].astype(str)
    n = len(gaia_ids)

    bprp0 = d['bprp0'] if 'bprp0' in keys else np.full(n, np.nan)

    if 'ages' in keys:
        ages = d['ages']
    else:
        ages = np.full(n, np.nan)
        if age_csv_paths:
            csv_list = age_csv_paths if isinstance(age_csv_paths, (list, tuple)) else [age_csv_paths]
            frames = []
            for p in csv_list:
                df_tmp = pd.read_csv(p)
                df_tmp['GaiaDR3_ID'] = df_tmp['GaiaDR3_ID'].astype(str)
                frames.append(df_tmp)
            df = pd.concat(frames, ignore_index=True).drop_duplicates(subset='GaiaDR3_ID', keep='first')
            if 'age_Myr' in df.columns:
                id_to_age = dict(zip(df['GaiaDR3_ID'], df['age_Myr']))
                ages = np.array([id_to_age.get(g, np.nan) for g in gaia_ids], dtype=float)
            if np.all(np.isnan(bprp0)) and 'BPRP0' in df.columns:
                id_to_bprp0 = dict(zip(df['GaiaDR3_ID'], df['BPRP0']))
                bprp0 = np.array([id_to_bprp0.get(g, np.nan) for g in gaia_ids], dtype=float)

    tic_ids = d['tic_ids'] if 'tic_ids' in keys else np.full(n, -1, dtype=np.int64)
    sectors = d['sectors'] if 'sectors' in keys else np.full(n, -1, dtype=np.int64)
    bprp0_err = d['bprp0_err'] if 'bprp0_err' in keys else np.full(n, np.nan)
    mg        = d['mg']        if 'mg'        in keys else np.full(n, np.nan)
    mg_err    = d['mg_err']    if 'mg_err'    in keys else np.full(n, np.nan)
    mem_prob  = d['mem_prob']  if 'mem_prob'  in keys else np.full(n, np.nan)

    n_with_age = int(np.sum(~np.isnan(ages)))
    print(f'Loaded latents cache ← {cache_path} ({n} samples, latent_dim={latent_vectors.shape[1]}, '
          f'ages: {n_with_age}/{n}{" — filled from CSV" if "ages" not in keys else ""})')
    if return_extras:
        return latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors, bprp0_err, mg, mg_err, mem_prob
    return latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors


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


def load_model(model_path: str, device: torch.device, hidden_size: int = 64,
               direction: str = 'bi', mode: str = 'parallel', use_flow: bool = False,
               num_meta_features: int = 0, use_conv_channels: bool = False,
               conv_config: dict = None):
    """Load trained model from checkpoint, auto-detecting architecture from state dict.

    The passed num_meta_features / use_conv_channels serve as hints but are overridden
    if the checkpoint's weights are inconsistent — e.g. a checkpoint trained without
    metadata will have no meta_encoder keys and a smaller input_proj.
    """
    ckpt = torch.load(model_path, map_location=device)
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

    # --- Auto-detect architecture from state dict ---
    has_meta = any(k.startswith('meta_encoder.') for k in sd)
    has_conv = any(k.startswith('ps_encoder.') for k in sd)

    if not has_meta and num_meta_features > 0:
        print(f'  [load_model] checkpoint has no meta_encoder; '
              f'overriding num_meta_features {num_meta_features} → 0')
        num_meta_features = 0
    if has_meta and num_meta_features == 0:
        # checkpoint has meta but caller said 0 — infer a nonzero placeholder so
        # meta_encoder is created; actual num_meta_features is stored on the model
        # after load_state_dict succeeds.  We can't know the original value from
        # weights alone so raise an informative error instead.
        raise ValueError(
            'Checkpoint has meta_encoder weights but num_meta_features=0 was passed. '
            'Re-run with the correct --num_meta_features used during training.'
        )
    if not has_conv and use_conv_channels:
        print(f'  [load_model] checkpoint has no conv encoders; '
              f'overriding use_conv_channels → False')
        use_conv_channels = False

    # Override from saved config if present (newer checkpoints only)
    meta_use_mask = False
    split_meta_encoders = False
    adversarial_sector_head = False
    num_sectors = 0
    instr_emb_dim = 16
    if isinstance(ckpt, dict):
        num_meta_features = ckpt.get('num_meta_features', num_meta_features)
        # Newer checkpoints store whether the metadata encoder uses an explicit
        # binary mask channel (DOROTHY-style masking). Old checkpoints lack the
        # key -> False, matching their use_mask=False meta encoder.
        meta_use_mask = ckpt.get('meta_use_mask', False)
        # Split-encoder / adversarial-head experiment flags (absent -> False/0).
        split_meta_encoders = ckpt.get('split_meta_encoders', False)
        adversarial_sector_head = ckpt.get('adversarial_sector_head', False)
        num_sectors = ckpt.get('num_sectors', 0)
        instr_emb_dim = ckpt.get('instr_emb_dim', 16)

    model = BiDirectionalMinGRU(
        hidden_size=hidden_size,
        direction=direction,
        mode=mode,
        use_flow=use_flow,
        num_meta_features=num_meta_features,
        use_conv_channels=use_conv_channels,
        conv_config=conv_config,
        meta_use_mask=meta_use_mask,
        split_meta_encoders=split_meta_encoders,
        instr_emb_dim=instr_emb_dim,
        adversarial_sector_head=adversarial_sector_head,
        num_sectors=num_sectors,
    ).to(device)

    model.load_state_dict(sd)
    model.eval()
    print(f'Loaded model from {model_path} '
          f'(hidden={hidden_size}, direction={direction}, num_meta={num_meta_features}, '
          f'meta_use_mask={meta_use_mask}, conv={use_conv_channels}, flow={use_flow})')
    return model


def compute_multiscale_features(h_valid: torch.Tensor, t_valid: torch.Tensor,
                                n_segments: int = 4,
                                minmax_edge_skip: int = 0,
                                diff_weight_mode: str = 'unweighted',
                                minmax_quantile: float = 0.0,
                                subtract_temporal_mean: bool = False,
                                glob_mode: str = 'voronoi',
                                seg_mode: str = 'equal_time',
                                hidden_size: int = None,
                                voronoi_cap_factor: float = 3.0) -> torch.Tensor:
    """Compute multi-scale temporal features from hidden states (time-aware).

    Pool operators are defined in physical time rather than sample index:
    global mean/std are weighted by Voronoi-style time duration per sample,
    segments split [t_min, t_max] into equal-time bins, and the difference
    aggregators operate on rates Δh/Δt rather than per-step Δh. This removes
    the sampling-rate / sector-length bias the original sample-indexed pool
    was injecting into the latent.

    Order statistics (global_max / global_min) default to plain extrema. With
    `minmax_quantile > 0` they switch to symmetric quantiles (e.g. 5th/95th)
    which are robust to single-step encoder spikes — useful when sectors with
    large mid-sector data gaps produce post-gap settling artifacts that the
    raw min/max picks up.

    Args:
        h_valid: (L, H) hidden states for valid timesteps.
        t_valid: (L,) physical time for each hidden state. Must be monotonic.
            Units are arbitrary but must match across the dataset (e.g. days).
        n_segments: number of equal-time segments for segment pooling.
        minmax_edge_skip: drop this many leading and trailing timesteps before
            computing global_min / global_max only. The MinGRU cumulative scan
            is barely averaged near each stream's start, so a single
            un-smoothed sample can dominate the per-feature min/max even after
            `trim_edges` strips raw input.
        diff_weight_mode: 'unweighted' (default; keeps the existing behavior
            of an unweighted mean over per-step rates Δh/Δt) or 'dt_inverse'
            (weight each rate sample by 1/Δt so dense within-sector pairs
            dominate the aggregate and the single rate sample spanning a
            large mid-sector gap is suppressed). 'dt_inverse' reduces to the
            unweighted mean for uniformly-sampled sequences.
        minmax_quantile: 0.0 (default) uses literal min/max. A value q in
            (0, 0.5) replaces them with the q-th and (1-q)-th sample-rank
            quantiles via torch.quantile. Stable against single-step outliers
            from post-gap settling in long sequences. Note: this is rank-based
            (unweighted) rather than time-weighted — at small q the percentile
            cut already absorbs outlier-step bias, and the unweighted form is
            ~10× faster than a per-feature time-weighted sort.
        subtract_temporal_mean: when True, subtract the Voronoi-time-weighted
            per-sample mean from h_valid before computing every block except
            diff_* (rates are already invariant to a constant offset). The
            glob_mean block then becomes exactly zero by construction and is
            emitted as a zero vector to preserve output dimensionality. This
            removes the L-drift signal that scales with sequence length —
            60-day sectors had glob_mean z≈+5σ relative to 27-day sectors
            because the MinGRU cumulative scan reaches a different equilibrium
            at 2× the timesteps. Centring the hidden state per sample erases
            that drift but keeps the *spread* and *shape* signals (glob_std,
            glob_max, glob_min, seg*, first_h, last_h become deviations from
            sample mean rather than absolute values).

    Returns:
        Feature vector of shape (12*H,):
        [glob_mean, glob_std, glob_max, glob_min,
         seg0..seg(n_segments-1), first_h, last_h, diff_mean, diff_std].
    """
    L, H = h_valid.shape
    device, dtype = h_valid.device, h_valid.dtype
    eps = torch.finfo(dtype).eps
    features = []

    # Per-sample weights for the time-average blocks (glob_mean/std + segments):
    #   'uniform'        : every sample weighted equally (sample-average).
    #   'voronoi'        : each sample carries half the gap to each neighbour
    #                      (endpoints get the inner half-gap) — its Voronoi time
    #                      cell, i.e. a true time-average of a piecewise-constant
    #                      trajectory. For regular cadence interior weights are
    #                      equal (endpoints get half).
    #   'voronoi_capped' : Voronoi, but each gap is clamped to
    #                      voronoi_cap_factor × median(Δt) first, so the two
    #                      samples bracketing a large gap can't dominate the mean.
    if glob_mode == 'uniform' or L == 1:
        w = torch.ones(L, device=device, dtype=dtype)
    elif glob_mode in ('voronoi', 'voronoi_capped'):
        dt = (t_valid[1:] - t_valid[:-1]).to(dtype).clamp_min(eps)   # (L-1,)
        if glob_mode == 'voronoi_capped':
            cap = (voronoi_cap_factor * dt.median()).clamp_min(eps)
            dt = dt.clamp_max(cap)
        w = torch.empty(L, device=device, dtype=dtype)
        w[0]  = 0.5 * dt[0]
        w[-1] = 0.5 * dt[-1]
        if L > 2:
            w[1:-1] = 0.5 * (dt[1:] + dt[:-1])
        w = w.clamp_min(0.0)
    else:
        raise ValueError(f"Unknown glob_mode: {glob_mode!r}")
    w_sum = w.sum().clamp_min(eps)
    w_col = w.unsqueeze(1)

    # 1. Global statistics — time-weighted.
    global_mean = (w_col * h_valid).sum(dim=0) / w_sum
    centered    = h_valid - global_mean.unsqueeze(0)
    global_var  = (w_col * centered.pow(2)).sum(dim=0) / w_sum
    global_std  = global_var.clamp_min(0.0).sqrt()

    # Optional centring: every downstream block (seg, first/last, glob_max/min)
    # then sees deviations from the per-sample weighted mean. diff_* uses
    # first differences and is already offset-invariant — no change needed.
    if subtract_temporal_mean:
        h_pool = centered
        # glob_mean is the quantity we just removed → emit zeros for it.
        global_mean = torch.zeros_like(global_mean)
    else:
        h_pool = h_valid

    m = max(0, int(minmax_edge_skip))
    if m > 0 and L > 2 * m + 1:
        h_inner = h_pool[m:L - m, :]
        w_inner = w[m:L - m]
    else:
        h_inner = h_pool
        w_inner = w

    q = float(minmax_quantile)
    if q > 0.0:
        qs = torch.tensor([q, 1.0 - q], device=device, dtype=dtype)
        quantiles = torch.quantile(h_inner, qs, dim=0)                # (2, H)
        global_min = quantiles[0]
        global_max = quantiles[1]
    else:
        global_max = h_inner.max(dim=0).values
        global_min = h_inner.min(dim=0).values
    features.extend([global_mean, global_std, global_max, global_min])

    # 2. Segment pooling. seg_mode selects how the n_segments bins are defined
    # and how empty bins are handled (segments reuse the `w` weights above, so
    # glob_mode controls the within-bin weighting too):
    #   'equal_time'       : equal-duration bins of [t_min, t_max]; an empty bin
    #                        emits a zero vector (legacy behaviour).
    #   'equal_time_carry' : equal-duration bins; an empty bin is filled with the
    #                        DIRECTION-AWARE carry of the bracketing states —
    #                        forward half from the last sample before the bin,
    #                        backward half from the first sample after it (the
    #                        state the model would assign to a point in the gap).
    #   'equal_count'      : equal-sample-count bins (the original quartile pool);
    #                        never empty, so no carry/zero-fill is needed.
    if seg_mode in ('equal_time', 'equal_time_carry'):
        t_min = t_valid[0]
        t_max = t_valid[-1]
        span  = (t_max - t_min).clamp_min(eps)
        edges = t_min + span * torch.linspace(0.0, 1.0, n_segments + 1,
                                              device=device, dtype=t_valid.dtype)
        # Forward/backward split for direction-aware carry: bidirectional latents
        # are [h_fwd | h_bwd] along the feature axis, so Hd = hidden_size.
        Hd = hidden_size if (hidden_size is not None and 2 * hidden_size == H) else None
        for seg_idx in range(n_segments):
            lo = edges[seg_idx]
            hi = edges[seg_idx + 1]
            # Closed on the right of the final bin so t_max is included.
            in_bin = (t_valid >= lo) & (t_valid < hi) if seg_idx < n_segments - 1 \
                     else (t_valid >= lo) & (t_valid <= hi)
            if in_bin.any():
                ws = w[in_bin]
                denom = ws.sum().clamp_min(eps)
                seg_mean = (ws.unsqueeze(1) * h_pool[in_bin, :]).sum(dim=0) / denom
            elif seg_mode == 'equal_time_carry':
                # a = last sample before the (interior) empty bin, b = a+1.
                a = int((t_valid < lo).sum().item()) - 1
                a = min(max(a, 0), L - 1)
                b = min(a + 1, L - 1)
                if Hd is not None:
                    seg_mean = torch.cat([h_pool[a, :Hd], h_pool[b, Hd:2 * Hd]], dim=0)
                else:
                    seg_mean = h_pool[a, :]   # single-direction fallback
            else:
                seg_mean = torch.zeros(H, device=device, dtype=dtype)
            features.append(seg_mean)
    elif seg_mode == 'equal_count':
        idx_edges = torch.linspace(0, L, n_segments + 1).round().to(torch.long)
        for seg_idx in range(n_segments):
            s = int(idx_edges[seg_idx]); e = int(idx_edges[seg_idx + 1])
            if e > s:
                ws = w[s:e]
                denom = ws.sum().clamp_min(eps)
                seg_mean = (ws.unsqueeze(1) * h_pool[s:e, :]).sum(dim=0) / denom
            else:
                seg_mean = torch.zeros(H, device=device, dtype=dtype)
            features.append(seg_mean)
    else:
        raise ValueError(f"Unknown seg_mode: {seg_mode!r}")

    # 3. First / last hidden states (anchored at t_min, t_max).
    features.append(h_pool[0, :])
    features.append(h_pool[-1, :])

    # 4. Temporal-derivative statistics — RATE Δh/Δt.
    # Each per-step pair (Δh_i, Δt_i) contributes one observation. The default
    # 'unweighted' mean treats every step equally; this still produces a
    # length-anomalous mean when a single rate sample spans a large mid-sector
    # gap because, even though that rate is bounded by 1/Δt, count-based mean
    # weights it as 1/L_pairs — heavy enough to shift the aggregate for stars
    # whose gap creates a coordinated Δh sign across features.
    # 'dt_inverse' weights each rate by 1/Δt, so dense within-sector pairs
    # dominate and the long-Δt boundary sample is suppressed by ~Δt_typ/Δt.
    if L > 1:
        dt_step = (t_valid[1:] - t_valid[:-1]).to(dtype).clamp_min(eps)
        rates = (h_valid[1:, :] - h_valid[:-1, :]) / dt_step.unsqueeze(1)
        if diff_weight_mode == 'dt':
            # Time-weighted: weight each per-step rate by its duration Δt. The
            # mean telescopes to the net slope Σ(Δh)/Σ(Δt) = (h_last-h_first)/span
            # (gap-robust); the std is the Δt-weighted dispersion of the rate
            # (quadratic-variation-per-time). Neither blows up on dense cadence.
            w_d = dt_step.unsqueeze(1)                                # (L-1, 1)
            w_d_sum = w_d.sum().clamp_min(eps)
            diff_mean = (w_d * rates).sum(dim=0) / w_d_sum
            if rates.shape[0] > 1:
                centered = rates - diff_mean.unsqueeze(0)
                diff_var = (w_d * centered.pow(2)).sum(dim=0) / w_d_sum
                diff_std = diff_var.clamp_min(0.0).sqrt()
            else:
                diff_std = torch.zeros(H, device=device, dtype=dtype)
        elif diff_weight_mode == 'dt_inverse':
            w_d = (1.0 / dt_step).unsqueeze(1)                        # (L-1, 1)
            w_d_sum = w_d.sum().clamp_min(eps)
            diff_mean = (w_d * rates).sum(dim=0) / w_d_sum
            if rates.shape[0] > 1:
                centered = rates - diff_mean.unsqueeze(0)
                diff_var = (w_d * centered.pow(2)).sum(dim=0) / w_d_sum
                diff_std = diff_var.clamp_min(0.0).sqrt()
            else:
                diff_std = torch.zeros(H, device=device, dtype=dtype)
        elif diff_weight_mode == 'unweighted':
            diff_mean = rates.mean(dim=0)
            diff_std  = rates.std(dim=0) if rates.shape[0] > 1 else torch.zeros(H, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown diff_weight_mode: {diff_weight_mode!r}")
        features.extend([diff_mean, diff_std])
    else:
        features.extend([torch.zeros(H, device=device, dtype=dtype),
                         torch.zeros(H, device=device, dtype=dtype)])
    
    return torch.cat(features, dim=0)


def extract_latent_vectors(model, h5_path: str, device: torch.device, max_length: int = None,
                           batch_size: int = 32, pooling_mode: str = 'multiscale',
                           sample_indices: np.ndarray = None, use_metadata: bool = False,
                           zero_metadata: bool = False,
                           zero_fields: list = None,
                           use_conv_channels: bool = False, trim_edges: int = 0,
                           minmax_edge_skip: int = 0,
                           diff_weight_mode: str = 'unweighted',
                           minmax_quantile: float = 0.0,
                           subtract_temporal_mean: bool = False,
                           apply_head_norm: bool = False,
                           pool_configs: list = None):
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
    metadata_features = METADATA_FEATURES

    # --- Load only the lightweight arrays upfront ---
    with h5py.File(h5_path, 'r') as f:
        h5_size = len(f['flux'])

        if sample_indices is not None:
            valid_mask = sample_indices < h5_size
            if not np.all(valid_mask):
                n_invalid = np.sum(~valid_mask)
                print(f'Warning: {n_invalid} sample indices exceed HDF5 size ({h5_size}). Filtering to valid indices.')
                sample_indices = sample_indices[valid_mask]
            if len(sample_indices) == 0:
                raise ValueError('No valid sample indices remain after filtering')
            sorted_indices = np.sort(sample_indices)
            lengths = f['length'][sorted_indices]
        else:
            sorted_indices = None
            lengths = f['length'][:]

        if trim_edges > 0:
            # Strip leading/trailing `trim_edges` samples from each light curve at
            # encoding time. Pipeline artifacts at e.g. raw index 7 (TESS scattered
            # light / thermal settling) leak into the multiscale latent via the GRU
            # hidden states; trimming them here produces clean re-encoded latents
            # without retraining the model.
            min_post = 32
            n_dropped = int(np.sum(lengths < 2 * trim_edges + min_post))
            if n_dropped:
                print(f'  trim_edges={trim_edges}: {n_dropped} light curves shorter '
                      f'than 2*trim+{min_post}; their post-trim length will be clamped to 0.')
            lengths = np.maximum(lengths - 2 * trim_edges, 0)

        # Metadata is small (one scalar per star per feature) — load it all at once
        if load_metadata and 'metadata' in f:
            meta_grp = f['metadata']
            n = len(sorted_indices) if sorted_indices is not None else h5_size
            raw = {}
            for feat in metadata_features:
                if feat in meta_grp:
                    vals = meta_grp[feat][sorted_indices] if sorted_indices is not None else meta_grp[feat][:]
                    raw[feat] = vals.astype(np.float32)
                else:
                    raw[feat] = np.zeros(n, dtype=np.float32)
            standardizer = MetadataStandardizer(fields=metadata_features)
            metadata_all = standardizer.transform(raw)
            # Validity mask (1 = present). Diagnostic ablations zero selected
            # fields' VALUES *and* their mask-channel entries — matching how the
            # model saw withheld metadata during DOROTHY-style training
            # (value 0 + mask 0). For a mask-trained encoder this keeps the
            # ablation in-distribution rather than presenting "a genuine 0".
            meta_mask_all = np.ones_like(metadata_all)
            zero_cols = []
            if zero_metadata:
                zero_cols = list(range(len(metadata_features)))
                print('  --zero_metadata: ALL metadata fields withheld (value 0, mask 0)')
            elif zero_fields:
                zero_cols = [metadata_features.index(f) for f in zero_fields
                             if f in metadata_features]
                bad = [f for f in zero_fields if f not in metadata_features]
                if bad:
                    print(f'  WARNING: --zero_fields entries not in metadata, ignored: {bad}')
                kept = [metadata_features[c] for c in zero_cols]
                print(f'  --zero_fields: withholding {kept} (value 0, mask 0)')
            if zero_cols:
                metadata_all[:, zero_cols] = 0.0
                meta_mask_all[:, zero_cols] = 0.0
        else:
            metadata_all = None
            meta_mask_all = None

    n_samples = len(lengths)
    latent_vectors = []
    # When pool_configs is given (multiscale only), emit one latent per config
    # from the SAME forward pass — the RNN cost is paid once.
    multi = {cfg['name']: [] for cfg in pool_configs} if pool_configs else None

    # Resolve spectra sidecar path for conv channels
    spectra_h5_path = None
    if use_conv_channels:
        stem = h5_path.rsplit('.h5', 1)[0]
        candidate = stem + '_spectra.h5'
        if Path(candidate).exists():
            spectra_h5_path = candidate
        else:
            with h5py.File(h5_path, 'r') as f_check:
                if 'power' in f_check and 'f_stat' in f_check and 'acf' in f_check:
                    spectra_h5_path = h5_path
        if spectra_h5_path is None:
            raise FileNotFoundError(
                f'Conv channel data not found. Expected sidecar at {candidate} '
                f'or power/f_stat/acf datasets inside {h5_path}.')
        print(f'  Conv channel source: {spectra_h5_path}')

    print(f'Extracting latent vectors for {n_samples} light curves (pooling={pooling_mode})...')

    # Keep the H5 file open and read flux/time batch-by-batch to avoid loading
    # the full (N × L) arrays into memory at once.
    spectra_ctx = h5py.File(spectra_h5_path, 'r') if (spectra_h5_path and spectra_h5_path != h5_path) else None
    with h5py.File(h5_path, 'r') as f, torch.no_grad():
        sf = spectra_ctx if spectra_ctx is not None else f
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            # Resolve the actual H5 row indices for this batch
            if sorted_indices is not None:
                batch_h5_idx = sorted_indices[start_idx:end_idx]
            else:
                batch_h5_idx = np.arange(start_idx, end_idx)

            # Pad to the longest actual (post-trim) sequence in this batch.
            lengths_batch = lengths[start_idx:end_idx]
            batch_max = int(lengths_batch.max())
            slc_start = trim_edges
            slc_stop  = trim_edges + batch_max
            flux_batch = torch.tensor(f['flux'][batch_h5_idx, slc_start:slc_stop], dtype=torch.float32, device=device)
            flux_err_batch = torch.tensor(f['flux_err'][batch_h5_idx, slc_start:slc_stop], dtype=torch.float32, device=device)
            time_batch = torch.tensor(f['time'][batch_h5_idx, slc_start:slc_stop], dtype=torch.float32, device=device)

            if use_conv_channels and spectra_h5_path is not None:
                conv_data_batch = {
                    'power':  torch.tensor(sf['power'][batch_h5_idx],  dtype=torch.float32, device=device),
                    'f_stat': torch.tensor(sf['f_stat'][batch_h5_idx], dtype=torch.float32, device=device),
                    'acf':    torch.tensor(sf['acf'][batch_h5_idx],    dtype=torch.float32, device=device),
                }
            else:
                conv_data_batch = None

            # Create mask (1 = valid, 0 = padded)
            B, L = flux_batch.shape
            mask = torch.zeros(B, L, device=device)
            for i, length in enumerate(lengths_batch):
                mask[i, :length] = 1.0

            x_in = torch.stack([flux_batch, flux_err_batch], dim=-1)  # (B, L, 2)
            t_in = time_batch.unsqueeze(-1)                            # (B, L, 1)

            meta_batch = (
                torch.tensor(metadata_all[start_idx:end_idx], dtype=torch.float32, device=device)
                if metadata_all is not None else None
            )
            meta_mask_batch = (
                torch.tensor(meta_mask_all[start_idx:end_idx], dtype=torch.float32, device=device)
                if meta_mask_all is not None else None
            )

            out = model(x_in, t_in, mask=mask, metadata=meta_batch, meta_mask=meta_mask_batch, conv_data=conv_data_batch, return_states=True)

            h_fwd = out.get('h_fwd_tensor')  # (B, L, H)
            h_bwd = out.get('h_bwd_tensor')  # (B, L, H)
            t_enc = out.get('t_enc')         # (B, L, num_time_enc_dims)

            if apply_head_norm:
                if not hasattr(model, 'head_norm') or model.head_norm is None:
                    raise ValueError("apply_head_norm=True but model has no head_norm module")
                if t_enc is None:
                    raise ValueError("apply_head_norm requires t_enc in model output")
                # Reconstruct the exact tensor head_norm was trained to normalize:
                # [h_fwd ‖ h_bwd ‖ t_enc] per timestep. Then slice off the hidden parts.
                H = model.hidden_size
                if h_fwd is not None and h_bwd is not None:
                    h_bi = torch.cat([h_fwd, h_bwd, t_enc], dim=-1)
                elif h_fwd is not None:
                    h_bi = torch.cat([h_fwd, t_enc], dim=-1)
                elif h_bwd is not None:
                    h_bi = torch.cat([h_bwd, t_enc], dim=-1)
                else:
                    raise ValueError("No hidden states returned by model")
                h_bi = model.head_norm(h_bi)
                if h_fwd is not None and h_bwd is not None:
                    h_fwd = h_bi[..., :H]
                    h_bwd = h_bi[..., H:2 * H]
                elif h_fwd is not None:
                    h_fwd = h_bi[..., :H]
                else:
                    h_bwd = h_bi[..., :H]

            for i in range(B):
                valid_len = lengths_batch[i]
                if h_fwd is not None and h_bwd is not None:
                    h_combined = torch.cat([h_fwd[i, :valid_len, :], h_bwd[i, :valid_len, :]], dim=-1)
                elif h_fwd is not None:
                    h_combined = h_fwd[i, :valid_len, :]
                elif h_bwd is not None:
                    h_combined = h_bwd[i, :valid_len, :]
                else:
                    raise ValueError("No hidden states returned by model")
                t_combined = time_batch[i, :valid_len]

                if pooling_mode == 'mean':
                    latent = h_combined.mean(dim=0)
                elif pooling_mode == 'final':
                    latent = h_combined[-1, :]
                elif pooling_mode == 'multiscale':
                    if multi is not None:
                        # Fan out all pool configs from the same hidden states.
                        for cfg in pool_configs:
                            lat = compute_multiscale_features(
                                h_combined, t_combined, n_segments=4,
                                minmax_edge_skip=minmax_edge_skip,
                                minmax_quantile=minmax_quantile,
                                subtract_temporal_mean=subtract_temporal_mean,
                                hidden_size=model.hidden_size,
                                glob_mode=cfg.get('glob_mode', 'voronoi'),
                                seg_mode=cfg.get('seg_mode', 'equal_time'),
                                diff_weight_mode=cfg.get('diff_mode', diff_weight_mode),
                            )
                            multi[cfg['name']].append(lat.cpu().numpy())
                        continue
                    latent = compute_multiscale_features(
                        h_combined, t_combined, n_segments=4,
                        minmax_edge_skip=minmax_edge_skip,
                        diff_weight_mode=diff_weight_mode,
                        minmax_quantile=minmax_quantile,
                        subtract_temporal_mean=subtract_temporal_mean,
                        hidden_size=model.hidden_size,
                    )
                else:
                    raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

                latent_vectors.append(latent.cpu().numpy())

            if (start_idx // batch_size) % 10 == 0:
                print(f'  Processed {end_idx}/{n_samples} light curves')
    
    if spectra_ctx is not None:
        spectra_ctx.close()

    if multi is not None:
        out = {name: np.stack(lst, axis=0) for name, lst in multi.items()}
        for name, arr in out.items():
            print(f'  [{name}] extracted {arr.shape}')
        return out

    latent_vectors = np.stack(latent_vectors, axis=0)
    print(f'Extracted latent vectors with shape: {latent_vectors.shape}')
    return latent_vectors


def load_ages(h5_path: str, csv_path: str, sample_indices: np.ndarray = None):
    """Load ages, BPRP0, gaia_ids, tic_ids, and sectors from the H5 file and phot_all.csv.

    Args:
        h5_path: path to H5 data file (used to read GaiaDR3_ID, tic, and sector per light curve)
        csv_path: path to phot_all.csv with stellar ages and BPRP0 keyed by GaiaDR3_ID
        sample_indices: specific indices to load (None = all)

    Returns 10-tuple: ages, bprp0, bprp0_err, mg, mg_err, mem_prob, gaia_ids,
    tic_ids, sectors. All metadata arrays are NaN where the CSV lookup fails.
    """
    # Read GaiaDR3_IDs, TIC IDs, and sectors from H5 metadata
    with h5py.File(h5_path, 'r') as f:
        raw_ids = f['metadata']['GaiaDR3_ID'][:]  # bytes strings
        tic_ids = f['metadata']['tic'][:]          # int64
        sectors = f['metadata']['sector'][:]       # int64

    if sample_indices is not None:
        sorted_indices = np.sort(sample_indices)
        raw_ids = raw_ids[sorted_indices]
        tic_ids = tic_ids[sorted_indices]
        sectors = sectors[sorted_indices]

    gaia_ids = np.array([g.decode('utf-8').strip() for g in raw_ids])

    # Defensive check: a precision-lossy float64 round-trip leaves Gaia IDs as
    # sci-notation strings ('1.234e+18'), which silently lose the join against
    # the CSV's proper int64 IDs. Warn loudly — the TIC fallback below recovers
    # metadata for these rows but ages/mem_prob (CSVs without TIC) are still lost.
    n_sci = sum('e' in g.lower() for g in gaia_ids)
    if n_sci:
        print(f'WARNING: {n_sci}/{len(gaia_ids)} GaiaDR3_IDs in {h5_path} are in '
              f'scientific notation (precision-lossy). Run scripts/fix_host_h5_gaia_ids.py '
              f'to repair the H5 in place.')

    # Load one or more CSV files and merge on GaiaDR3_ID.
    # Earlier files take priority for duplicate entries (e.g. metadata.csv ages
    # override host_all_metadata.csv which has no ages).
    csv_paths = csv_path if isinstance(csv_path, list) else [csv_path]
    frames = []
    for p in csv_paths:
        df_tmp = pd.read_csv(p)
        df_tmp['GaiaDR3_ID'] = df_tmp['GaiaDR3_ID'].astype(str)
        frames.append(df_tmp)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset='GaiaDR3_ID', keep='first')

    id_to_age       = dict(zip(df['GaiaDR3_ID'], df['age_Myr']))       if 'age_Myr'      in df.columns else {}
    id_to_bprp0     = dict(zip(df['GaiaDR3_ID'], df['BPRP0']))         if 'BPRP0'        in df.columns else {}
    id_to_bprp0_err = dict(zip(df['GaiaDR3_ID'], df['BPRP0_err']))     if 'BPRP0_err'    in df.columns else {}
    id_to_mg        = dict(zip(df['GaiaDR3_ID'], df['MG_quick']))      if 'MG_quick'     in df.columns else {}
    id_to_mg_err    = dict(zip(df['GaiaDR3_ID'], df['MG_quick_err']))  if 'MG_quick_err' in df.columns else {}
    id_to_mem_prob  = dict(zip(df['GaiaDR3_ID'], df['mem_prob_val']))  if 'mem_prob_val' in df.columns else {}

    # TIC-based fallback maps (only for CSVs that carry TIC_ID, e.g. host_all_metadata.csv).
    # Used to fill rows whose Gaia ID didn't match — robust to sci-notation corruption.
    df_tic = df[df.get('TIC_ID', pd.Series(dtype=int)).notna()] if 'TIC_ID' in df.columns else None
    tic_to_bprp0     = dict(zip(df_tic['TIC_ID'].astype('int64'), df_tic['BPRP0']))         if df_tic is not None and 'BPRP0'        in df.columns else {}
    tic_to_bprp0_err = dict(zip(df_tic['TIC_ID'].astype('int64'), df_tic['BPRP0_err']))     if df_tic is not None and 'BPRP0_err'    in df.columns else {}
    tic_to_mg        = dict(zip(df_tic['TIC_ID'].astype('int64'), df_tic['MG_quick']))      if df_tic is not None and 'MG_quick'     in df.columns else {}
    tic_to_mg_err    = dict(zip(df_tic['TIC_ID'].astype('int64'), df_tic['MG_quick_err']))  if df_tic is not None and 'MG_quick_err' in df.columns else {}

    ages      = np.array([id_to_age.get(gid, np.nan)       for gid       in gaia_ids],          dtype=float)
    bprp0     = np.array([id_to_bprp0.get(gid, tic_to_bprp0.get(int(t), np.nan))         for gid, t in zip(gaia_ids, tic_ids)], dtype=float)
    bprp0_err = np.array([id_to_bprp0_err.get(gid, tic_to_bprp0_err.get(int(t), np.nan)) for gid, t in zip(gaia_ids, tic_ids)], dtype=float)
    mg        = np.array([id_to_mg.get(gid, tic_to_mg.get(int(t), np.nan))               for gid, t in zip(gaia_ids, tic_ids)], dtype=float)
    mg_err    = np.array([id_to_mg_err.get(gid, tic_to_mg_err.get(int(t), np.nan))       for gid, t in zip(gaia_ids, tic_ids)], dtype=float)
    mem_prob  = np.array([id_to_mem_prob.get(gid, np.nan)  for gid       in gaia_ids],          dtype=float)

    n_with_age = np.sum(~np.isnan(ages))
    print(f'Loaded ages for {n_with_age}/{len(ages)} light curves ({100*n_with_age/len(ages):.1f}%)')

    return ages, bprp0, bprp0_err, mg, mg_err, mem_prob, gaia_ids, tic_ids, sectors


def plot_umap(latent_vectors: np.ndarray, ages: np.ndarray, output_path: str,
              n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'euclidean',
              bprp0_min: float = None, bprp0_max: float = None, bprp0: np.ndarray = None,
              gaia_ids: np.ndarray = None, tic_ids: np.ndarray = None, sectors: np.ndarray = None):
    """Create UMAP embedding and plot colored by stellar age.

    UMAP is fit on ALL samples (including those without ages) so the embedding
    structure reflects the full dataset. The plot shows only stars with valid
    ages (since coloring requires age); the saved npz keeps all rows.
    """
    import umap  # Lazy import to avoid requiring umap for non-plotting scripts

    n_total = len(latent_vectors)
    n_with_age = int(np.sum(~np.isnan(ages)))
    print(f'Computing UMAP on all {n_total} samples ({n_with_age} have valid ages)...')

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=42,
        verbose=True
    )
    embedding = reducer.fit_transform(latent_vectors)

    print(f'UMAP embedding shape: {embedding.shape}')

    # Plot: include all stars (BPRP0-filtered if requested). Stars without
    # ages are drawn first as a grey backdrop; stars with ages are overlaid
    # and colored by log10(age).
    bprp_mask = np.ones(n_total, dtype=bool)
    if bprp0 is not None:
        if bprp0_min is not None:
            bprp_mask &= (bprp0 >= bprp0_min)
        if bprp0_max is not None:
            bprp_mask &= (bprp0 <= bprp0_max)

    age_mask = ~np.isnan(ages)
    grey_mask = bprp_mask & ~age_mask
    color_mask = bprp_mask & age_mask
    n_excluded_bprp = n_total - int(bprp_mask.sum())
    print(f'Plotting {int(color_mask.sum())} aged + {int(grey_mask.sum())} unaged stars'
          f'{f" (excluded {n_excluded_bprp} outside BPRP0 range)" if n_excluded_bprp else ""}')

    fig, ax = plt.subplots(figsize=(12, 10))

    # Grey backdrop for stars without ages.
    if grey_mask.any():
        ax.scatter(
            embedding[grey_mask, 0],
            embedding[grey_mask, 1],
            c='lightgrey', s=20, alpha=0.4, linewidths=0,
            label=f'no age (n={int(grey_mask.sum())})',
        )

    ages_plot = ages[color_mask]
    scatter = ax.scatter(
        embedding[color_mask, 0],
        embedding[color_mask, 1],
        c=ages_plot,
        cmap='viridis',
        norm=LogNorm(vmin=ages_plot.min(), vmax=ages_plot.max()),
        s=50, alpha=0.7,
    )

    cbar = plt.colorbar(scatter, ax=ax, label='Stellar Age (Myr)')
    cbar.ax.set_ylabel('Stellar Age (Myr)', fontsize=12)

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP of Light Curve Latent Space\nColored by Stellar Age', fontsize=14)
    if grey_mask.any():
        ax.legend(loc='lower left', fontsize=10)

    textstr = (f'N (aged): {int(color_mask.sum())}\n'
               f'N (no age): {int(grey_mask.sum())}\n'
               f'Age range: {ages_plot.min():.1f} - {ages_plot.max():.1f} Myr')
    if bprp0_min is not None or bprp0_max is not None:
        bprp0_str = f'BPRP0: {bprp0_min if bprp0_min is not None else "—"} to {bprp0_max if bprp0_max is not None else "—"}'
        textstr += f'\n{bprp0_str}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved UMAP plot to {output_path}')
    
    # Save the FULL embedding data (all rows, before any BPRP0/age filtering)
    # in a single npz so downstream notebooks can replot with arbitrary cuts
    # without recomputing UMAP. The high-dim latents are NOT saved here — load
    # them from the per-source --save_latents cache if needed.
    npz_path = output_path.replace('.png', '_data.npz')
    save_dict = dict(
        embedding=embedding,
        ages=ages,
        bprp0=bprp0 if bprp0 is not None else np.array([]),
    )
    if gaia_ids is not None:
        save_dict['gaia_ids'] = gaia_ids.astype(str)
    if tic_ids is not None:
        save_dict['tic_ids'] = tic_ids
    if sectors is not None:
        save_dict['sectors'] = sectors
    np.savez(npz_path, **save_dict)
    print(f'Saved embedding data to {npz_path}')

    return embedding, ages


def main():
    parser = argparse.ArgumentParser(description='UMAP visualization of latent space colored by stellar age')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint. Required unless --load_latents is set.')
    parser.add_argument('--h5_path', type=str, nargs='+', default=['data/timeseries_x.h5'], help='Path(s) to H5 data file(s)')
    parser.add_argument('--age_csv_path', type=str, nargs='+', default=['data/metadata.csv'], help='Path(s) to CSV file(s) with stellar ages keyed by GaiaDR3_ID (e.g. metadata.csv host_all_metadata.csv)')
    parser.add_argument('--output_dir', type=str, default='output/simple_rnn/umap_plots', help='Output directory')
    parser.add_argument('--hidden_size', type=int, default=64, help='Model hidden size')
    parser.add_argument('--direction', type=str, default='bi', choices=['forward', 'backward', 'bi'])
    parser.add_argument('--mode', type=str, default='parallel', choices=['sequential', 'parallel'])
    parser.add_argument('--use_flow', action='store_true')
    parser.add_argument('--use_metadata', action='store_true')
    parser.add_argument('--zero_metadata', action='store_true',
                        help='Diagnostic: zero the standardized metadata tensor before '
                             'encoding (requires --use_metadata). Withholds the metadata '
                             'signal while keeping the meta-encoder dimensionally intact. '
                             'Used to probe whether the latent encodes sector via the '
                             'light curve alone (Route B) vs. via metadata (Route A).')
    parser.add_argument('--zero_fields', type=str, default=None,
                        help='Comma-separated metadata field names to withhold (value 0 + '
                             'mask 0), keeping all others — e.g. "sector,camera,ccd" to drop '
                             'the instrumental confounds while keeping astrophysical metadata. '
                             'Requires --use_metadata; ignored if --zero_metadata is set.')
    parser.add_argument('--use_conv_channels', action='store_true')
    parser.add_argument('--trim_edges', type=int, default=0,
                        help='If >0, strip this many samples from each end of every light '
                             'curve before encoding (removes leading/trailing pipeline artifacts). '
                             'Only applies to fresh extraction, not --load_latents.')
    parser.add_argument('--minmax_edge_skip', type=int, default=0,
                        help='If >0, drop this many leading/trailing hidden-state timesteps '
                             'before computing global_min / global_max only. Mitigates the '
                             'MinGRU cumulative-scan edge bias on per-feature extrema. '
                             'Mean/std/segment/diff stats are unaffected.')
    parser.add_argument('--diff_weight_mode', type=str, default='unweighted',
                        choices=['unweighted', 'dt_inverse'],
                        help="How to aggregate per-step rates Δh/Δt in the diff_mean/diff_std "
                             "blocks. 'unweighted' (default) is the existing equal-count mean. "
                             "'dt_inverse' weights each rate sample by 1/Δt so dense within-sector "
                             "pairs dominate and the long-Δt rate sample spanning a mid-sector "
                             "gap is suppressed.")
    parser.add_argument('--minmax_quantile', type=float, default=0.0,
                        help='If >0, replace global_min / global_max with time-weighted '
                             'quantiles at q and 1-q (e.g. 0.05 -> 5th/95th percentile). '
                             'Robust against single-step encoder spikes near data-gap boundaries.')
    parser.add_argument('--subtract_temporal_mean', action='store_true',
                        help='Subtract per-sample Voronoi-time-weighted mean from hidden '
                             'states before pooling. Removes the L-dependent drift in '
                             'glob_mean / seg / first_h / last_h that caused 60-day sectors '
                             '(s97/s98) to form an isolated UMAP island. diff_* are unchanged '
                             '(rates are offset-invariant); glob_mean output becomes zero.')
    parser.add_argument('--apply_head_norm', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply model.head_norm (the LayerNorm seen by the reconstruction '
                             'head during pretraining) to the per-timestep hidden states '
                             'before pooling. Restores train/extract consistency: '
                             'pretraining normalizes [h_fwd ‖ h_bwd ‖ t_enc] per-timestep '
                             'before the head, but the return_states=True path returns raw '
                             'unnormalized hidden states. With this flag we apply head_norm '
                             'to the same 136-D vector and slice off the first 2*H features. '
                             'NOTE: this is per-timestep feature-norm only; it does not address '
                             'sequence-length drift. Use --no-apply_head_norm to disable.')
    parser.add_argument('--conv_encoder_type', type=str, default='unet', choices=['lightweight', 'unet'])
    parser.add_argument('--max_length', type=int, default=2048, help='Max sequence length for inference')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--n_neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP min_dist parameter')
    parser.add_argument('--pooling_mode', type=str, default='multiscale',
                        choices=['mean', 'multiscale', 'final'],
                        help='How to aggregate hidden states into a latent vector')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process. None = use all.')
    parser.add_argument('--stratify_by_age', action='store_true',
                        help='Sample evenly across log-age bins instead of taking first N samples')
    parser.add_argument('--n_age_bins', type=int, default=20,
                        help='Number of age bins for stratified sampling')
    parser.add_argument('--bprp0_min', type=float, default=None,
                        help='Minimum BPRP0 value for plot filtering (post-UMAP). None = no cut.')
    parser.add_argument('--bprp0_max', type=float, default=None,
                        help='Maximum BPRP0 value for plot filtering (post-UMAP). None = no cut.')
    parser.add_argument('--load_latents', type=str, nargs='+', default=None, metavar='PATH',
                        help='One or more cached latent npz files (from this script or '
                             'kfold_age_inference.py). When set, skips model loading and '
                             'extraction entirely. Caches are concatenated in order. '
                             '--model_path and H5 reads are bypassed.')
    parser.add_argument('--save_latents', type=str, nargs='+', default=None, metavar='PATH',
                        help='Save latents to these paths (one per --h5_path or per '
                             '--load_latents, in order). In cache mode this re-saves each '
                             'loaded cache in the canonical kfold format (e.g. ages filled '
                             'from CSV, key renamed latents→latent_vectors).')
    args = parser.parse_args()
    if args.load_latents is None and args.model_path is None:
        parser.error('--model_path is required unless --load_latents is set.')
    n_sources = len(args.load_latents) if args.load_latents else len(args.h5_path)
    src_name = '--load_latents' if args.load_latents else '--h5_path'
    if args.save_latents is not None and len(args.save_latents) != n_sources:
        parser.error(f'--save_latents has {len(args.save_latents)} paths but {src_name} '
                     f'has {n_sources}; provide one save path per source.')
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each H5 file (or each cache), then concatenate results
    all_latents, all_ages, all_bprp0, all_gaia_ids, all_tic_ids, all_sectors = [], [], [], [], [], []

    if args.load_latents is not None:
        # Cache mode: skip model loading and H5 extraction entirely.
        for i, cache_path in enumerate(args.load_latents):
            lv, a, b, g, t, s, be, mgv, mge, mp = load_latents_cache(
                cache_path, age_csv_paths=args.age_csv_path, return_extras=True)
            if args.save_latents is not None:
                save_latents_cache(args.save_latents[i], lv, a, b, g, t, s,
                                   bprp0_err=be, mg=mgv, mg_err=mge, mem_prob=mp)
            all_latents.append(lv)
            all_ages.append(a); all_bprp0.append(b)
            all_gaia_ids.append(g); all_tic_ids.append(t); all_sectors.append(s)
    else:
        num_meta_features = len(METADATA_FEATURES) if args.use_metadata else 0

        conv_config = None
        if args.use_conv_channels:
            conv_config = {
                'encoder_type': args.conv_encoder_type,
                'input_length': 16000,
                'encoder_dims': [4, 8, 16, 32],
                'num_layers': 4 if args.conv_encoder_type == 'unet' else 3,
                'hidden_channels': 16,
                'activation': 'gelu',
            }

        model = load_model(
            args.model_path, device,
            hidden_size=args.hidden_size,
            direction=args.direction,
            mode=args.mode,
            use_flow=args.use_flow,
            num_meta_features=num_meta_features,
            use_conv_channels=args.use_conv_channels,
            conv_config=conv_config,
        )

        for i, h5_path in enumerate(args.h5_path):
            print(f'\n--- Processing {h5_path} ---')

            # Determine sample indices for this file
            sample_indices = None
            if args.max_samples is not None:
                ages_tmp, *_ = load_ages(h5_path, args.age_csv_path, sample_indices=None)
                if args.stratify_by_age:
                    sample_indices = get_stratified_indices(
                        ages_tmp, max_samples=args.max_samples, n_bins=args.n_age_bins)
                else:
                    valid_indices = np.where(~np.isnan(ages_tmp))[0]
                    sample_indices = valid_indices[:args.max_samples]
                with h5py.File(h5_path, 'r') as f:
                    h5_size = len(f['flux'])
                valid_mask = sample_indices < h5_size
                if not np.all(valid_mask):
                    print(f'Warning: filtering {np.sum(~valid_mask)} out-of-range indices')
                    sample_indices = sample_indices[valid_mask]

            latent_vectors = extract_latent_vectors(
                model, h5_path, device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                pooling_mode=args.pooling_mode,
                sample_indices=sample_indices,
                use_metadata=args.use_metadata,
                zero_metadata=args.zero_metadata,
                zero_fields=(args.zero_fields.split(',') if args.zero_fields else None),
                use_conv_channels=args.use_conv_channels,
                trim_edges=args.trim_edges,
                minmax_edge_skip=args.minmax_edge_skip,
                diff_weight_mode=args.diff_weight_mode,
                minmax_quantile=args.minmax_quantile,
                subtract_temporal_mean=args.subtract_temporal_mean,
                apply_head_norm=args.apply_head_norm,
            )

            ages, bprp0, bprp0_err, mg, mg_err, mem_prob, gaia_ids, tic_ids, sectors = load_ages(
                h5_path, args.age_csv_path, sample_indices=sample_indices)

            if args.save_latents is not None:
                save_latents_cache(args.save_latents[i],
                                   latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors,
                                   bprp0_err=bprp0_err, mg=mg, mg_err=mg_err, mem_prob=mem_prob)

            all_latents.append(latent_vectors)
            all_ages.append(ages); all_bprp0.append(bprp0)
            all_gaia_ids.append(gaia_ids); all_tic_ids.append(tic_ids); all_sectors.append(sectors)

    latent_vectors = np.concatenate(all_latents, axis=0)
    ages     = np.concatenate(all_ages)
    bprp0    = np.concatenate(all_bprp0)
    gaia_ids = np.concatenate(all_gaia_ids)
    tic_ids  = np.concatenate(all_tic_ids)
    sectors  = np.concatenate(all_sectors)
    n_sources = len(args.load_latents) if args.load_latents else len(args.h5_path)
    src_label = 'cache(s)' if args.load_latents else 'H5 file(s)'
    print(f'\nTotal: {len(latent_vectors)} light curves across {n_sources} {src_label}')

    # Generate output filename
    if args.model_path is not None:
        model_name = Path(args.model_path).stem
    else:
        model_name = Path(args.load_latents[0]).stem
    suffix = f'{args.pooling_mode}'
    if args.bprp0_min is not None or args.bprp0_max is not None:
        bprp_range = f'bprp{args.bprp0_min or ""}-{args.bprp0_max or ""}'
        suffix += f'_{bprp_range}'
    output_path = output_dir / f'umap_age_{model_name}_{suffix}.png'

    plot_umap(
        latent_vectors, ages, str(output_path),
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        bprp0_min=args.bprp0_min,
        bprp0_max=args.bprp0_max,
        bprp0=bprp0,
        gaia_ids=gaia_ids,
        tic_ids=tic_ids,
        sectors=sectors,
    )

    print('Done!')


if __name__ == '__main__':
    main()
