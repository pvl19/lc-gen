"""
Train hierarchical RNN (minLSTM/minGRU) autoencoder on light curve time series.

Uses parallelizable RNN variants from "Were RNNs All We Needed?" (arXiv:2410.01201)
Supports training with block masking and time-aware positional encoding.
"""

import numpy as np
import math
import argparse
import sys
from pathlib import Path
from functools import partial
from tqdm import tqdm
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lcgen.models.rnn import HierarchicalRNN, RNNConfig
from lcgen.data.masking import dynamic_block_mask


class LightCurveDataset(Dataset):
    """Dataset for light curve time series with timestamps."""

    def __init__(self, flux, timestamps):
        """
        Args:
            flux: Numpy array of shape (N, seq_len)
            timestamps: Numpy array of shape (N, seq_len)
        """
        self.flux = torch.from_numpy(flux).float()
        self.timestamps = torch.from_numpy(timestamps).float()
        self.flux_err = None

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        # Return flux and timestamps - masking will be applied in collate_fn
        item = {
            'flux': self.flux[idx],
            'time': self.timestamps[idx]
        }
        if self.flux_err is not None:
            item['flux_err'] = self.flux_err[idx]
        return item


def collate_with_masking(batch, min_block_size=1, max_block_size=None,
                         min_mask_ratio=0.1, max_mask_ratio=0.9):
    """
    Custom collate function that applies masking to light curves.

    Creates input as [masked_flux, mask_indicator] for the transformer.
    """
    # Extract flux and timestamps
    batch_flux = torch.stack([item['flux'] for item in batch], dim=0)  # [batch_size, seq_len]
    batch_time = torch.stack([item['time'] for item in batch], dim=0)  # [batch_size, seq_len]

    batch_size, seq_len = batch_flux.shape

    if max_block_size is None:
        max_block_size = seq_len // 2

    # Apply masking per sample
    batch_inputs = []
    batch_targets = []
    batch_masks = []
    batch_block_sizes = []
    batch_mask_ratios = []

    # Determine whether flux_err exists in items and stack once
    flux_err_exists = 'flux_err' in batch[0]
    if flux_err_exists:
        batch_flux_err = torch.stack([item['flux_err'] for item in batch], dim=0)
    else:
        batch_flux_err = None

    for i in range(batch_size):
        flux = batch_flux[i]
        # Get per-sample flux_err (or zeros)
        if batch_flux_err is not None:
            flux_err = batch_flux_err[i]
        else:
            flux_err = torch.zeros_like(flux)

        # Apply dynamic block masking to flux
        flux_masked, mask, block_size, mask_ratio = dynamic_block_mask(
            flux,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            min_mask_ratio=min_mask_ratio,
            max_mask_ratio=max_mask_ratio
        )

        # Stack channels as: [masked_flux, flux_err, mask_indicator]
        # Shape: (seq_len, 3)
        input_with_mask = torch.stack([flux_masked, flux_err, mask.float()], dim=-1)

        batch_inputs.append(input_with_mask)
        batch_targets.append(flux)
        batch_masks.append(mask)
        batch_block_sizes.append(block_size)
        batch_mask_ratios.append(mask_ratio)

    return {
        'input': torch.stack(batch_inputs),  # (batch, seq_len, 3)
        'target': torch.stack(batch_targets),  # (batch, seq_len)
        'time': batch_time,  # (batch, seq_len)
        'mask': torch.stack(batch_masks),  # (batch, seq_len)
        'block_size': torch.tensor(batch_block_sizes),
        'mask_ratio': torch.tensor(batch_mask_ratios)
    }


def load_lightcurve_data(hdf5_file):
    """
    Load light curve data from HDF5 file.

    Expected structure:
        - 'flux': (N, seq_len) - normalized flux values
        - 'time': (N, seq_len) - timestamps

    Args:
        hdf5_file: Path to HDF5 file

    Returns:
        flux: Numpy array of shape (N, seq_len)
        timestamps: Numpy array of shape (N, seq_len)
    """
    with h5py.File(hdf5_file, 'r') as f:
        flux = f['flux'][:]
        timestamps = f['time'][:]
        # Optional flux_err dataset
        flux_err = None
        if 'flux_err' in f:
            flux_err = f['flux_err'][:]
        print(f"Loaded flux: {flux.shape}, time: {timestamps.shape}")
        if flux_err is not None:
            print(f"Loaded flux_err: {flux_err.shape}; min={float(np.nanmin(flux_err)):.3e}, max={float(np.nanmax(flux_err)):.3e}, mean={float(np.nanmean(flux_err)):.3e}")
        else:
            raise RuntimeError(
                "HDF5 file is missing required dataset 'flux_err'. "
                "Please recreate the HDF5 from the source 'star_sector_lc_formatted.pickle' so that per-observation measurement errors are included."
            )

        # Check for NaNs
        n_nans_flux = np.sum(np.isnan(flux))
        n_nans_time = np.sum(np.isnan(timestamps))
        if n_nans_flux > 0:
            print(f"Warning: {n_nans_flux} NaN values found in flux")
            flux = np.nan_to_num(flux, nan=0.0)
        if n_nans_time > 0:
            print(f"Warning: {n_nans_time} NaN values found in timestamps")
            timestamps = np.nan_to_num(timestamps, nan=0.0)

    return flux, timestamps, flux_err


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, train_mc_samples=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_masked_loss = 0
    total_unmasked_loss = 0
    n_batches = 0

    for batch in dataloader:
        x_input = batch['input'].to(device)  # (batch, seq_len, 3) [flux, flux_err, mask]
        x_flux_err = x_input[..., 1]
        x_target = batch['target'].to(device)  # (batch, seq_len)
        timestamps = batch['time'].to(device)  # (batch, seq_len)
        mask = batch['mask'].to(device)  # (batch, seq_len)

        # Forward pass
        optimizer.zero_grad()
        # Pass target and mask to support autoregressive teacher forcing when enabled
        output = model(x_input, timestamps, target=x_target, mask=mask)
        # New training loss: compute negative log-probability of the observed
        # flux under the conditional flow posterior for each timestep.
        # No samples are drawn from the flow during training; instead we use
        # the flow's density evaluated at the observed target. This loss is
        # computed over all timesteps (both masked and unmasked) as requested.
        B, L = x_target.shape
        if not (isinstance(output, dict) and output.get('flow_context', None) is not None and getattr(model, 'flow_posterior', None) is not None):
            raise RuntimeError('train_epoch: flow posterior required for log-prob training loss (missing flow_context or model.flow_posterior)')

        flow_ctx = output['flow_context']  # (B, L, C)
        ctx_flat = flow_ctx.view(B * L, -1)
        # Observations shaped as (B*L, 1) to match flow event shape
        y_flat = x_target.view(B * L, 1)

        # Compute conditional distribution and evaluate log-prob at the observed targets
        dist = model.flow_posterior(ctx_flat)
        # dist.log_prob should return (B*L, 1) or (B*L,) depending on implementation
        logp = dist.log_prob(y_flat)
        # Ensure shape is (B*L,)
        if logp.dim() > 1:
            logp = logp.view(-1)
        logp = logp.view(B, L)

        # Negative log-likelihood averaged over all timesteps
        loss = -logp.mean()

        # For logging split masked/unmasked negative-log-likelihoods
        masked_nll = (-logp)[mask].mean() if mask.any() else torch.tensor(0.0, device=device)
        unmasked_nll = (-logp)[~mask].mean() if (~mask).any() else torch.tensor(0.0, device=device)

    # (masked_nll and unmasked_nll already computed per-branch above)

        # Backprop and optimize on full NLL loss
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Step scheduler after each batch (for OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_masked_loss += masked_nll.item()
        total_unmasked_loss += unmasked_nll.item()
        n_batches += 1

    if n_batches == 0:
        return {'total_loss': 0.0, 'masked_loss': 0.0, 'unmasked_loss': 0.0}

    return {
        'total_loss': total_loss / n_batches,
        'masked_loss': total_masked_loss / n_batches,
        'unmasked_loss': total_unmasked_loss / n_batches
    }


def validate_epoch(model, dataloader, criterion, device, mc_samples=32):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_masked_loss = 0
    total_unmasked_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x_input = batch['input'].to(device)
            x_target = batch['target'].to(device)
            timestamps = batch['time'].to(device)
            mask = batch['mask'].to(device)

            # Forward pass
            # Provide target and mask so the model can use teacher forcing in autoregressive mode
            output = model(x_input, timestamps, target=x_target, mask=mask)

            if isinstance(output, dict) and 'flow_context' in output and output['flow_context'] is not None:
                # Flow context available: evaluate flow log-prob at observed targets.
                # The flow is expected to be conditioned on measurement error
                # through the model's context, so we use the flow density directly
                # (no convolution with measurement error).
                flow_ctx = output['flow_context']  # (B, L, C)
                B, L = x_target.shape
                ctx_flat = flow_ctx.view(B * L, -1)
                y_flat = x_target.view(B * L, 1)
                if getattr(model, 'flow_posterior', None) is None:
                    raise RuntimeError('Model provided flow_context but no flow_posterior module found')

                dist = model.flow_posterior(ctx_flat)
                logp = dist.log_prob(y_flat)
                if logp.dim() > 1:
                    logp = logp.view(-1)
                logp = logp.view(B, L)

                loss = -logp.mean()
                masked_loss = (-logp)[mask].mean() if mask.any() else torch.tensor(0.0, device=device)
                unmasked_loss = (-logp)[~mask].mean() if (~mask).any() else torch.tensor(0.0, device=device)
            else:
                recon = output['reconstructed']
                pred_mean = recon[..., 0]
                pred_logsigma = recon[..., 1]
                pred_sigma = torch.nn.functional.softplus(pred_logsigma)

                var = pred_sigma ** 2 + 1e-8
                nll = 0.5 * (torch.log(2 * math.pi * var) + (pred_mean - x_target) ** 2 / var)

                loss = nll.mean()
                masked_loss = nll[mask].mean() if mask.any() else torch.tensor(0.0, device=device)
                unmasked_loss = nll[~mask].mean() if (~mask).any() else torch.tensor(0.0, device=device)

            total_loss += loss.item()
            total_masked_loss += masked_loss.item()
            total_unmasked_loss += unmasked_loss.item()
            n_batches += 1

    if n_batches == 0:
        return {'total_loss': 0.0, 'masked_loss': 0.0, 'unmasked_loss': 0.0}

    return {
        'total_loss': total_loss / n_batches,
        'masked_loss': total_masked_loss / n_batches,
        'unmasked_loss': total_unmasked_loss / n_batches
    }


def plot_reconstruction_examples(model, dataloader, device, save_path, n_examples=4, sample_eval=True, num_samples=3, base_seed=0, eval_teacher_forcing=False, save_examples_data: bool = False):
    """Plot reconstruction examples. If sample_eval is True, generate multiple sampled
    reconstructions per example (saved as separate files)."""
    model.eval()

    # Collect up to `n_examples` examples from the dataloader. This allows the
    # plotting routine to display 4 examples even when the dataloader's batch
    # size is smaller than 4 by concatenating multiple batches.
    inputs_list = []
    targets_list = []
    times_list = []
    masks_list = []
    block_sizes_list = []
    mask_ratios_list = []

    collected = 0
    data_iter = iter(dataloader)
    while collected < n_examples:
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        bs = batch['input'].shape[0]
        need = n_examples - collected
        take = min(need, bs)

        inputs_list.append(batch['input'][:take])
        targets_list.append(batch['target'][:take])
        times_list.append(batch['time'][:take])
        masks_list.append(batch['mask'][:take])
        block_sizes_list.append(batch['block_size'][:take])
        mask_ratios_list.append(batch['mask_ratio'][:take])

        collected += take

    if collected == 0:
        print("No data available from dataloader to plot examples.")
        return

    # Stack and ensure we only keep up to n_examples (in case more were collected)
    x_input = torch.cat(inputs_list, dim=0)[:n_examples].to(device)
    x_target = torch.cat(targets_list, dim=0)[:n_examples].to(device)
    timestamps = torch.cat(times_list, dim=0)[:n_examples].to(device)
    mask = torch.cat(masks_list, dim=0)[:n_examples].to(device)
    block_sizes = torch.cat(block_sizes_list, dim=0)[:n_examples].numpy()
    mask_ratios = torch.cat(mask_ratios_list, dim=0)[:n_examples].numpy()

    with torch.no_grad():
        # Base reconstruction: either pass target/mask (teacher forcing) or not
        if eval_teacher_forcing:
            base_out = model(x_input, timestamps, target=x_target, mask=mask)
        else:
            base_out = model(x_input, timestamps)

        if isinstance(base_out, dict) and 'reconstructed' in base_out:
            recon = base_out['reconstructed']
            recon_mean = recon[..., 0]
            recon_logsigma = recon[..., 1]
            recon_sigma = torch.nn.functional.softplus(recon_logsigma)
            sampled_initial = base_out.get('sampled', None)
        else:
            recon = base_out
            recon_mean = recon[..., 0]
            recon_sigma = torch.nn.functional.softplus(recon[..., 1])
            sampled_initial = None

        sampled_list = []
        if sample_eval:
            # Prefer drawing samples directly from the decoder-level flow context
            # when available. This produces S samples from p_flow(·|flow_ctx)
            # and is consistent with the conditional flow used during training.
            if isinstance(base_out, dict) and base_out.get('flow_context', None) is not None and getattr(model, 'flow_posterior', None) is not None:
                flow_ctx = base_out['flow_context']  # (B, L, C)
                B, L, C = flow_ctx.shape
                ctx_flat = flow_ctx.view(B * L, -1)
                dist = model.flow_posterior(ctx_flat)
                # Try to sample the whole flattened context at once
                try:
                    s_all = dist.sample((num_samples,))  # (S, B*L, 1) or (S, B*L)
                    s_all = s_all.view(num_samples, B, L)
                except TypeError:
                    # Some flow implementations are not indexable for batched sampling
                    # or may not accept a generator; sample per-context (slower)
                    flat_list = []
                    for idx in range(B * L):
                        ctx_i = ctx_flat[idx:idx+1]
                        dist_i = model.flow_posterior(ctx_i)
                        si = dist_i.sample((num_samples,)).view(num_samples)
                        flat_list.append(si)
                    s_all = torch.stack(flat_list, dim=1).view(num_samples, B, L)

                # store per-sample tensors in sampled_list (on CPU for plotting)
                for s in range(num_samples):
                    sampled_list.append(s_all[s].cpu())
            else:
                # Fallback: call model per-seed (previous behavior)
                for s in range(num_samples):
                    seed = int(base_seed + s + 1)
                    out_s = model(x_input, timestamps, target=x_target, mask=mask, seed=seed)
                    if isinstance(out_s, dict):
                        sampled_list.append(out_s.get('sampled', None))
                    else:
                        sampled_list.append(None)

        # (example data saving moved to after conversion to numpy below)

    # Move to CPU for plotting
    x_target = x_target.cpu().numpy()
    recon_mean_np = recon_mean.cpu().numpy()
    recon_sigma_np = recon_sigma.cpu().numpy()
    timestamps_np = timestamps.cpu().numpy()
    mask_np = mask.cpu().numpy()
    # Also pull flux_err from inputs so we can save it with examples
    try:
        flux_err_np = x_input[..., 1].cpu().numpy()
    except Exception:
        flux_err_np = None

    # Now optionally save raw example data (targets, recon mean/sigma, mask, timestamps, samples)
    # This must happen after tensors are moved to CPU and converted to numpy arrays.
    if save_examples_data and sample_eval:
        try:
            from pathlib import Path as _P
            save_base = _P(str(save_path)).with_suffix('')
            npz_path = _P(f"{save_base}_examples.npz")

            # Build samples array: (num_samples, n_examples, seq_len)
            if len(sampled_list) > 0 and sampled_list[0] is not None:
                samples_np = np.stack([s.cpu().numpy() for s in sampled_list], axis=0)
            else:
                # Fallback: generate samples from mean+sigma using numpy RNG with base_seed
                samples_np = np.empty((num_samples, recon_mean_np.shape[0], recon_mean_np.shape[1]), dtype=float)
                for s in range(num_samples):
                    seed = int(base_seed + s + 1)
                    rng = np.random.default_rng(seed)
                    for i in range(recon_mean_np.shape[0]):
                        eps = rng.standard_normal(size=recon_mean_np.shape[1])
                        samples_np[s, i, :] = recon_mean_np[i] + recon_sigma_np[i] * eps

            # Save arrays: target, recon_mean, recon_sigma, mask, timestamps, samples
            np.savez_compressed(
                str(npz_path),
                target=x_target,
                recon_mean=recon_mean_np,
                recon_sigma=recon_sigma_np,
                flux_err=flux_err_np,
                mask=mask_np,
                timestamps=timestamps_np,
                samples=samples_np
            )
            print(f"Saved reconstruction example data to {npz_path}")
        except Exception as _e:
            print(f"Warning: failed to save example data: {_e}")

    if not sample_eval:
        fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3*n_examples))
        if n_examples == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            t = timestamps_np[i]
            target = x_target[i]
            recon_mean_i = recon_mean_np[i]
            recon_sigma_i = recon_sigma_np[i]
            m = mask_np[i].astype(bool)
            block_size = block_sizes[i]
            mask_ratio = mask_ratios[i]

            # Shade masked and unmasked intervals to make regions obvious
            # We'll compute contiguous segments for masked/unmasked and axvspan them
            def shade_regions(ax, t, mask_bool):
                """Shade only masked (True) contiguous segments without gaps.

                Uses t[idx] as the end boundary so adjacent segments touch.
                """
                labeled_mask = False
                n = len(mask_bool)
                if n == 0:
                    return
                cur = mask_bool[0]
                start = 0
                for idx in range(1, n):
                    if mask_bool[idx] != cur:
                        if cur:
                            x0 = float(t[start])
                            x1 = float(t[idx])
                            ax.axvspan(x0, x1, color='red', alpha=0.10, zorder=0, label='Masked region' if not labeled_mask else None)
                            labeled_mask = True
                        start = idx
                        cur = mask_bool[idx]
                # last segment
                if cur:
                    x0 = float(t[start])
                    x1 = float(t[-1])
                    ax.axvspan(x0, x1, color='red', alpha=0.10, zorder=0, label='Masked region' if not labeled_mask else None)

            shade_regions(ax, t, m)

            # Plot target thicker so it's obvious
            ax.plot(t, target, color='k', alpha=0.9, label='Target', linewidth=3.0, zorder=5)
            ax.plot(t, recon_mean_i, 'b-', alpha=0.85, label='Reconstruction (mean)', linewidth=1.5, zorder=4)
            ax.legend(loc='upper right', fontsize='small')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            ax.set_title(f'Example {i+1} (masked {mask_ratio*100:.1f}%, block size {block_size})')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved reconstruction plot to {save_path}")
        return

    # sample_eval True: create a single consolidated figure with one row per example.
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]

    def shade_regions(ax, t, mask_bool):
        """Shade only masked (True) contiguous segments without gaps.

        Uses t[idx] as the end boundary so adjacent segments touch.
        """
        labeled_mask = False
        n = len(mask_bool)
        if n == 0:
            return
        cur = mask_bool[0]
        start = 0
        for idx in range(1, n):
            if mask_bool[idx] != cur:
                if cur:
                    x0 = float(t[start])
                    x1 = float(t[idx])
                    ax.axvspan(x0, x1, color='red', alpha=0.10, zorder=0, label='Masked region' if not labeled_mask else None)
                    labeled_mask = True
                start = idx
                cur = mask_bool[idx]
        # last segment
        if cur:
            x0 = float(t[start])
            x1 = float(t[-1])
            ax.axvspan(x0, x1, color='red', alpha=0.10, zorder=0, label='Masked region' if not labeled_mask else None)

    for i, ax in enumerate(axes):
        t = timestamps_np[i]
        target = x_target[i]
        recon_mean_i = recon_mean_np[i]
        recon_sigma_i = recon_sigma_np[i]
        m = mask_np[i].astype(bool)
        block_size = block_sizes[i]
        mask_ratio = mask_ratios[i]

        # Shade masked/observed regions
        shade_regions(ax, t, m)

        # Plot target thicker so it's obvious
        ax.plot(t, target, color='k', alpha=0.95, label='Target', linewidth=3.0, zorder=5)
        ax.plot(t, recon_mean_i, 'b--', alpha=0.9, label='Mean reconstruction', linewidth=1.5, zorder=4)

        # Overlay samples (use model-provided samples when available, otherwise draw from mean+sigma)
        for s_idx in range(num_samples):
            s_arr = sampled_list[s_idx] if s_idx < len(sampled_list) else None
            if s_arr is not None:
                sample_vals = s_arr.cpu().numpy()[i]
            else:
                seed = int(base_seed + s_idx + 1)
                rng = np.random.default_rng(seed)
                eps = rng.standard_normal(size=recon_mean_i.shape)
                sample_vals = recon_mean_i + recon_sigma_i * eps

            ax.plot(t, sample_vals, alpha=0.7, label=f'Sample {s_idx+1}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Flux')
        ax.set_title(f'Example {i+1} (masked {mask_ratio*100:.1f}%, block {block_size})')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved reconstruction plot to {save_path}")

    # --- Zoomed-in view: for each example show a close-up of ~100 timesteps ---
    # Use a fixed window length (100 timesteps or seq_len if smaller). Center the
    # window on the largest masked segment (or sequence center if none).
    zoom_windows = []
    zoom_centers = []
    seq_len = timestamps_np.shape[1]
    window_len = min(100, seq_len)
    for i in range(n_examples):
        mask_bool = mask_np[i].astype(bool)
        # find contiguous True runs and pick the longest
        best_start, best_end, best_len = None, None, 0
        cur_start = None
        for j in range(seq_len):
            if mask_bool[j]:
                if cur_start is None:
                    cur_start = j
            else:
                if cur_start is not None:
                    l = j - cur_start
                    if l > best_len:
                        best_start, best_end, best_len = cur_start, j, l
                    cur_start = None
        # tail
        if cur_start is not None:
            l = seq_len - cur_start
            if l > best_len:
                best_start, best_end, best_len = cur_start, seq_len, l

        if best_start is None:
            # fallback: center window
            center = seq_len // 2
        else:
            center = (best_start + best_end) // 2

        s = max(0, center - window_len // 2)
        e = s + window_len
        if e > seq_len:
            e = seq_len
            s = max(0, e - window_len)
        zoom_windows.append((s, e))
        zoom_centers.append(center)

    # Create zoomed figure
    zoom_path = Path(str(save_path)).with_suffix('')
    zoom_path = Path(f"{zoom_path}_zoom.png")
    fig_z, axes_z = plt.subplots(n_examples, 1, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes_z = [axes_z]

    for i, ax in enumerate(axes_z):
        t = timestamps_np[i]
        recon_mean_i = recon_mean_np[i]
        recon_sigma_i = recon_sigma_np[i]
        start, end = zoom_windows[i]
        t_win = t[start:end]
        mean_win = recon_mean_i[start:end]
        sigma_win = recon_sigma_i[start:end]
        target_win = x_target[i][start:end]

        # shade masked area in this window
        mask_seg = mask_np[i][start:end].astype(bool)
        if mask_seg.any():
            # shade the entire masked points range (contiguous) for visibility
            seg_idx = np.where(mask_seg)[0]
            ax.axvspan(float(t_win[seg_idx[0]]), float(t_win[seg_idx[-1]]), color='red', alpha=0.12, zorder=0)

        ax.plot(t_win, target_win, color='k', linewidth=3.0, label='Target')
        ax.plot(t_win, mean_win, 'b--', linewidth=1.5, label='Mean')

        # plot samples
        for s_idx in range(num_samples):
            s_arr = sampled_list[s_idx] if s_idx < len(sampled_list) else None
            if s_arr is not None:
                sample_vals = s_arr.cpu().numpy()[i][start:end]
            else:
                seed = int(base_seed + s_idx + 1)
                rng = np.random.default_rng(seed)
                eps = rng.standard_normal(size=mean_win.shape)
                sample_vals = mean_win + sigma_win * eps
            ax.plot(t_win, sample_vals, alpha=0.8, label=f'Sample {s_idx+1}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Flux')
        ax.set_title(f'Zoom Example {i+1} (t[{start}:{end}])')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(zoom_path, dpi=150)
    plt.close()
    print(f"Saved zoomed reconstruction plot to {zoom_path}")

    # Also save individual zoom figures for each example
    window_lengths = [100, 200, 400]
    for i in range(n_examples):
        center = zoom_centers[i]
        seq_len = timestamps_np.shape[1]

        # Arrange zoom panels in a single column (one row per zoom length)
        fig_i, axes_i = plt.subplots(len(window_lengths), 1, figsize=(8, 3 * len(window_lengths)))
        if len(window_lengths) == 1:
            axes_i = [axes_i]

        for ax_idx, win_len in enumerate(window_lengths):
            s = max(0, center - win_len // 2)
            e = s + win_len
            if e > seq_len:
                e = seq_len
                s = max(0, e - win_len)

            t = timestamps_np[i][s:e]
            mean_win = recon_mean_np[i][s:e]
            target_win = x_target[i][s:e]
            sigma_win = recon_sigma_np[i][s:e]

            ax = axes_i[ax_idx]
            # Shade masked region within this window if present
            mask_seg = mask_np[i][s:e].astype(bool)
            if mask_seg.any():
                seg_idx = np.where(mask_seg)[0]
                ax.axvspan(float(t[seg_idx[0]]), float(t[seg_idx[-1]]), color='red', alpha=0.12, zorder=0)

            ax.plot(t, target_win, color='k', linewidth=3.0, label='Target')
            ax.plot(t, mean_win, 'b--', linewidth=1.5, label='Mean')

            for s_idx in range(num_samples):
                s_arr = sampled_list[s_idx] if s_idx < len(sampled_list) else None
                if s_arr is not None:
                    sample_vals = s_arr.cpu().numpy()[i][s:e]
                else:
                    seed = int(base_seed + s_idx + 1)
                    rng = np.random.default_rng(seed)
                    eps = rng.standard_normal(size=mean_win.shape)
                    sample_vals = mean_win + sigma_win * eps
                ax.plot(t, sample_vals, alpha=0.8, label=f'Sample {s_idx+1}')

            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            ax.set_title(f'Zoom {win_len} pts (t[{s}:{e}])')
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)

        per_zoom_path = Path(str(save_path)).with_suffix('')
        per_zoom_path = Path(f"{per_zoom_path}_example{i+1}_zoom.png")
        plt.tight_layout()
        plt.savefig(per_zoom_path, dpi=150)
        plt.close()
        print(f"Saved per-example zoom plot to {per_zoom_path}")


def main():
    parser = argparse.ArgumentParser(description='Train hierarchical RNN (minLSTM/minGRU) autoencoder on light curves')
    parser.add_argument('--input', type=str,
                        default='data/real_lightcurves/timeseries.h5',
                        help='Path to HDF5 file with light curve time series')
    parser.add_argument('--output_dir', type=str, default='models/rnn',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--encoder_dims', type=int, nargs='+', default=[64, 128],
                        help='Encoder dimensions for hierarchical levels (default: 2 levels)')
    parser.add_argument('--rnn_type', type=str, default='minGRU', choices=['minlstm', 'minGRU'],
                        help='RNN cell type (minlstm or minGRU)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of hierarchical levels')
    parser.add_argument('--num_layers_per_level', type=int, default=2,
                        help='RNN layers per hierarchy level')
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                        help='Maximum masking ratio')
    parser.add_argument('--block_size', type=int, default=32,
                        help='Maximum block size for masking')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation set fraction')
    parser.add_argument('--subset_size', type=int, default=0,
                        help='If >0, randomly sample this many examples from the dataset for quick tests')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--sample_eval', action='store_true',
                        help='When set, generate sampled reconstructions during evaluation/plotting')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of stochastic samples to generate per example when --sample_eval is set')
    parser.add_argument('--eval_mc_samples', type=int, default=32,
                        help='Number of Monte Carlo samples to approximate convolution of flow with measurement noise during validation')
    parser.add_argument('--train_mc_samples', type=int, default=0,
                        help='If >0, use MC convolution with this many samples during training for flow-based likelihood (slower)')
    parser.add_argument('--eval_teacher_forcing', action='store_true',
                        help='When set, use teacher forcing (pass target+mask) during evaluation and plotting; otherwise sample autoregressively')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint .pt file to resume training')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Normalize rnn_type argument to canonical values used by the codebase.
    # Accept case-insensitive inputs but map to 'minlstm' or 'minGRU'.
    rnn_raw = args.rnn_type
    if isinstance(rnn_raw, str) and rnn_raw.lower().startswith('mingru'):
        args.rnn_type = 'minGRU'
    elif isinstance(rnn_raw, str) and rnn_raw.lower().startswith('minlstm'):
        args.rnn_type = 'minlstm'
    else:
        # fallback to default 'minGRU'
        args.rnn_type = 'minGRU'

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data from {args.input}...")
    flux, timestamps, flux_err = load_lightcurve_data(args.input)

    # Optionally subsample the dataset for quick tests
    if args.subset_size and args.subset_size > 0:
        n_total = flux.shape[0]
        subset = min(args.subset_size, n_total)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n_total, size=subset, replace=False)
        flux = flux[idx]
        timestamps = timestamps[idx]
        if flux_err is not None:
            flux_err = flux_err[idx]

    print(f"\nDataset statistics:")
    print(f"  Shape: {flux.shape}")
    print(f"  Flux range: [{np.min(flux):.3f}, {np.max(flux):.3f}]")
    print(f"  Flux mean: {np.mean(flux):.3f}")
    print(f"  Flux std: {np.std(flux):.3f}")
    print(f"  Time range: [{np.min(timestamps):.3f}, {np.max(timestamps):.3f}]")

    # Create dataset
    dataset = LightCurveDataset(flux, timestamps)
    # Attach flux_err to dataset (as torch tensor). load_lightcurve_data now
    # enforces presence of 'flux_err' and will raise if missing, so here we
    # simply convert to torch tensor.
    dataset.flux_err = torch.from_numpy(flux_err).float()

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"\nDataset split:")
    print(f"  Train: {train_size} samples")
    print(f"  Val: {val_size} samples")

    # Create collate function with masking parameters
    collate_fn = partial(
        collate_with_masking,
        min_block_size=1,
        max_block_size=args.block_size,
        min_mask_ratio=0.1,
        max_mask_ratio=args.mask_ratio
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    seq_len = flux.shape[1]
    print(f"\nCreating Hierarchical RNN ({args.rnn_type}) autoencoder...")
    config = RNNConfig(
        input_dim=3,  # [flux, flux_err, mask]
        input_length=seq_len,
        encoder_dims=args.encoder_dims,
        rnn_type=args.rnn_type,
        num_layers_per_level=args.num_layers_per_level,
        dropout=0.0,  # No dropout
        min_period=0.00278,
        max_period=28
    )

    model = HierarchicalRNN(config).to(device)

    # Optionally resume from checkpoint
    start_epoch = 0
    if args.resume_checkpoint is not None:
        ckpt_path = Path(args.resume_checkpoint)
        if ckpt_path.exists():
            print(f"Loading checkpoint from {ckpt_path} to resume training...")
            ckpt = torch.load(ckpt_path, map_location=device)
            # Load model weights
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            # We'll recreate optimizer below and restore its state if available
            start_epoch = int(ckpt.get('epoch', 0)) + 1
            best_val_loss = float(ckpt.get('val_loss', best_val_loss if 'best_val_loss' in locals() else float('inf')))
            print(f"Resuming from epoch {start_epoch}; best_val_loss={best_val_loss}")
        else:
            print(f"Resume checkpoint {ckpt_path} not found; starting from scratch.")

    # --- Sanity check: single forward pass to verify input shapes / device placement ---
    try:
        sample_batch = next(iter(train_loader))
        x_input_sample = sample_batch['input'].to(device)
        timestamps_sample = sample_batch['time'].to(device)
        print("Sanity check - sample input shapes:", x_input_sample.shape, timestamps_sample.shape)
        # Provide sample target and mask for sanity-check forward pass (supports autoregressive mode)
        sample_out = model(
            x_input_sample,
            timestamps_sample,
            target=sample_batch['target'].to(device),
            mask=sample_batch['mask'].to(device)
        )
        if isinstance(sample_out, dict) and 'reconstructed' in sample_out:
            print("Sanity check forward OK. Reconstructed shape:", sample_out['reconstructed'].shape)
        else:
            print("Sanity check forward returned unexpected type:", type(sample_out))
    except Exception as e:
        import traceback
        print("Sanity check forward FAILED. Traceback:")
        traceback.print_exc()
        # Re-raise so the user sees the original error and training stops
        raise

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # OneCycleLR scheduler: only create if there are >0 total steps
    total_steps = len(train_loader) * args.epochs
    if total_steps > 0:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=1e2,
            final_div_factor=1e2,
            anneal_strategy='cos',
            # AdamW doesn't use momentum like SGD - disable cycling momentum
            cycle_momentum=False
        )
    else:
        scheduler = None

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    max_block_display = args.block_size if args.block_size else "seq_len//2"
    print(f"Dynamic masking: block size [1, {max_block_display}], "
          f"mask ratio [10%, {args.mask_ratio*100:.0f}%]")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scheduler, train_mc_samples=args.train_mc_samples)
        print("Training metrics:", train_metrics)
        train_losses.append(train_metrics['total_loss'])

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, mc_samples=args.eval_mc_samples)
        val_losses.append(val_metrics['total_loss'])

        # Print progress (match MLP format)
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['total_loss']:.6f} | "
              f"Masked: {train_metrics['masked_loss']:.6f} | "
              f"Unmasked: {train_metrics['unmasked_loss']:.6f}")
        print(f"  Val   - Loss: {val_metrics['total_loss']:.6f} | "
              f"Masked: {val_metrics['masked_loss']:.6f} | "
              f"Unmasked: {val_metrics['unmasked_loss']:.6f}")

        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            checkpoint_path = output_dir / 'rnn_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['total_loss'],
                'val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")

        # Plot samples every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            plot_path = output_dir / f'rnn_recon_v4_epoch{epoch+1}.png'
            # Request up to 4 examples for plotting; if the validation set is
            # smaller than 4, the plotting function will clip accordingly.
            plot_reconstruction_examples(
                model, val_loader, device, plot_path,
                n_examples=4,
                sample_eval=args.sample_eval,
                num_samples=args.num_samples,
                base_seed=args.seed,
                eval_teacher_forcing=args.eval_teacher_forcing
            )

        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'config': config
            }, output_dir / f'rnn_checkpoint_epoch{epoch+1}.pt')

    # Save final model
    final_path = output_dir / 'rnn_final.pt'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, final_path)
    print(f"\nSaved final model to {final_path}")

    # Always produce final-epoch reconstructions and save example data so user
    # can inspect and re-plot offline. This ensures the final epoch is plotted
    # even when epoch % 10 != 0.
    final_plot_path = output_dir / f'rnn_recon_v4_epoch{args.epochs}_final.png'
    plot_reconstruction_examples(
        model, val_loader, device, final_plot_path,
        n_examples=4,
        sample_eval=args.sample_eval,
        num_samples=args.num_samples,
        base_seed=args.seed,
        eval_teacher_forcing=args.eval_teacher_forcing,
        save_examples_data=True
    )

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Hierarchical RNN ({args.rnn_type}) Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'rnn_training_curve_v4.png', dpi=150)
    plt.close()

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
