"""Data preparation utilities for convolutional models (power spectra and ACFs).

Functions for normalizing power spectra and autocorrelation functions,
and for creating block masks for training.
"""
import numpy as np
import torch


def normalize_ps(ps, scale_factor=50):
    """Normalize power spectrum using log transformation.

    Args:
        ps: Power spectrum array
        scale_factor: Scaling factor for normalization (default 50)

    Returns:
        Normalized power spectrum
    """
    # Log transform and scale
    ps_log = np.log10(ps + 1e-10)  # Add small epsilon to avoid log(0)
    ps_norm = ps_log / scale_factor
    return ps_norm


def normalize_acf(acf, scale_factor=0.001):
    """Normalize autocorrelation function.

    Args:
        acf: Autocorrelation function array
        scale_factor: Scaling factor for normalization (default 0.001)

    Returns:
        Normalized ACF
    """
    # Simple scaling
    acf_norm = acf / scale_factor
    return acf_norm


def block_mask(data, block_size=50, mask_ratio=0.5):
    """Create random block masks for data.

    Args:
        data: Input data tensor/array of shape (batch, length) or (length,)
        block_size: Size of each masked block
        mask_ratio: Fraction of data to mask

    Returns:
        masked_data: Data with blocks set to zero
        mask: Boolean mask (True = masked, False = unmasked)
    """
    if isinstance(data, torch.Tensor):
        data = data.clone()
        is_tensor = True
    else:
        data = np.copy(data)
        is_tensor = False

    # Handle 1D or 2D input
    if data.ndim == 1:
        data = data[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    batch_size, length = data.shape
    mask = np.zeros((batch_size, length), dtype=bool)

    for i in range(batch_size):
        num_blocks = int((length * mask_ratio) / block_size)
        for _ in range(num_blocks):
            start_idx = np.random.randint(0, max(1, length - block_size))
            end_idx = min(start_idx + block_size, length)
            mask[i, start_idx:end_idx] = True
            if is_tensor:
                data[i, start_idx:end_idx] = 0
            else:
                data[i, start_idx:end_idx] = 0

    if squeeze:
        data = data.squeeze(0)
        mask = mask.squeeze(0)

    return data, mask


def block_predefined_mask(data, block_size=50, mask_ratio=0.5):
    """Create random block masks for data (alternative implementation).

    Similar to block_mask but returns mask where True=masked.

    Args:
        data: Input data tensor of shape (batch, length) or (length,)
        block_size: Size of each masked block
        mask_ratio: Fraction of data to mask

    Returns:
        masked_data: Data with blocks set to zero
        mask: Boolean mask (True = masked, False = unmasked)
    """
    if isinstance(data, torch.Tensor):
        masked_data = data.clone()
        device = data.device
    else:
        masked_data = torch.from_numpy(np.copy(data))
        device = 'cpu'

    # Handle 1D or 2D input
    if masked_data.dim() == 1:
        masked_data = masked_data.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    batch_size, length = masked_data.shape
    mask = torch.zeros((batch_size, length), dtype=torch.bool, device=device)

    for i in range(batch_size):
        total_to_mask = int(length * mask_ratio)
        masked_so_far = 0

        while masked_so_far < total_to_mask:
            # Random block size around the specified size
            current_block = min(block_size, total_to_mask - masked_so_far)
            start_idx = np.random.randint(0, max(1, length - current_block))
            end_idx = start_idx + current_block

            mask[i, start_idx:end_idx] = True
            masked_data[i, start_idx:end_idx] = 0
            masked_so_far += current_block

    if squeeze:
        masked_data = masked_data.squeeze(0)
        mask = mask.squeeze(0)

    return masked_data, mask
