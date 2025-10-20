"""Masking strategies for training autoencoders"""

import numpy as np
import torch


def block_predefined_mask(x, block_size=10, mask_ratio=0.7):
    """
    Create a predefined block mask for 1D or 2D power spectrum/ACF data.

    Randomly selects contiguous blocks to mask by setting them to zero.

    Args:
        x: Tensor of shape (seq_len,) or (channels, seq_len)
        block_size: Length of each masked block
        mask_ratio: Fraction of sequence to mask

    Returns:
        masked_x: Input with masked blocks set to 0
        mask: Boolean tensor, True where input is masked
    """
    seq_len = x.shape[-1]
    num_blocks = int(np.ceil(seq_len / block_size))
    num_masked = int(num_blocks * mask_ratio)

    masked_blocks = np.random.choice(np.arange(num_blocks), size=num_masked, replace=False)

    x = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool)

    for block in masked_blocks:
        start = block * block_size
        end = np.min([start + block_size, seq_len])

        if x.dim() == 1:
            x[start:end] = 0.0
            mask[start:end] = True
        else:  # (channels, seq_len)
            x[:, start:end] = 0.0
            mask[:, start:end] = True

    return x, mask


def block_predefined_mask_lc(x, block_size=10, mask_ratio=0.7):
    """
    Create a predefined block mask for light curve data with flux and uncertainty.

    For masked regions:
    - Flux is set to 0
    - Uncertainty is scaled by 16x (to indicate low confidence)

    Args:
        x: Tensor of shape (seq_len, 2) or (batch, seq_len, 2)
           where channels are [flux, flux_err]
        block_size: Length of each masked block
        mask_ratio: Fraction of sequence to mask

    Returns:
        x_masked: Input with masked flux and scaled uncertainty
        mask: Boolean tensor of shape (seq_len,) or (batch, seq_len),
              True where input is masked
    """
    # Handle single vs batch input
    single_input = False
    if x.dim() == 2:
        # (seq_len, 2) -> (1, seq_len, 2)
        x = x.unsqueeze(0)
        single_input = True

    batch_size, seq_len, channels = x.shape
    assert channels == 2, "Input must have 2 channels (flux, flux_err)"

    x_masked = x.clone()
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=x.device)

    for i in range(batch_size):
        num_blocks = int(np.ceil(seq_len / block_size))
        num_masked = int(num_blocks * mask_ratio)
        masked_blocks = np.random.choice(np.arange(num_blocks), size=num_masked, replace=False)

        for block in masked_blocks:
            start = block * block_size
            end = min(start + block_size, seq_len)
            x_masked[i, start:end, 0] = 0.0  # Flux -> 0
            x_masked[i, start:end, 1] = 16 * x_masked[i, start:end, 1]  # Uncertainty -> 16x
            mask[i, start:end] = True

    if single_input:
        return x_masked[0], mask[0]
    else:
        return x_masked, mask


def block_mask(x, block_size=100, num_blocks=1):
    """
    Mask random contiguous blocks by setting them to 0.

    This is a simpler alternative to block_predefined_mask that specifies
    the number of blocks rather than a mask ratio.

    Args:
        x: Tensor of shape (seq_len,) or (channels, seq_len)
        block_size: Length of each masked block
        num_blocks: Number of blocks to mask

    Returns:
        masked_x: Input with masked blocks
        mask: Boolean tensor, True where input is kept, False where masked
    """
    x = x.clone()
    mask = torch.ones_like(x, dtype=torch.bool)
    seq_len = x.shape[-1]

    for _ in range(num_blocks):
        start = np.random.randint(0, max(1, seq_len - block_size))
        end = min(seq_len, start + block_size)

        if x.dim() == 1:
            x[start:end] = 0.0
            mask[start:end] = False
        else:  # (channels, seq_len)
            x[:, start:end] = 0.0
            mask[:, start:end] = False

    return x, mask


def random_point_mask(x, mask_ratio=0.15):
    """
    Create random point-wise mask (BERT-style masking).

    Args:
        x: Tensor of any shape
        mask_ratio: Fraction of points to mask

    Returns:
        masked_x: Input with masked points set to 0
        mask: Boolean tensor, True where input is masked
    """
    mask = torch.rand_like(x) < mask_ratio
    masked_x = x.clone()
    masked_x[mask] = 0.0
    return masked_x, mask


def create_random_block_mask_batch(batch_size, seq_len, mask_ratio=0.15, block_size=5):
    """
    Create random block masks for a batch of sequences.

    Args:
        batch_size: Number of sequences
        seq_len: Length of each sequence
        mask_ratio: Fraction of sequence to mask
        block_size: Size of contiguous blocks to mask

    Returns:
        mask: Boolean tensor of shape (batch_size, seq_len),
              True for positions to mask
    """
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Calculate number of blocks to mask
    n_masked = int(seq_len * mask_ratio)
    n_blocks = max(1, n_masked // block_size)

    # Randomly select block starting positions for each sequence
    for i in range(batch_size):
        possible_starts = list(range(0, seq_len - block_size + 1))
        if len(possible_starts) > 0 and n_blocks > 0:
            block_starts = np.random.choice(
                possible_starts,
                size=min(n_blocks, len(possible_starts)),
                replace=False
            )
            # Mask blocks
            for start in block_starts:
                mask[i, start:start + block_size] = True

    return mask


def dynamic_block_mask(x, min_block_size=1, max_block_size=None,
                       min_mask_ratio=0.1, max_mask_ratio=0.9):
    """
    Create dynamic block mask with random block sizes and mask ratios.

    At each call, randomly samples:
    - Block size from uniform distribution [min_block_size, max_block_size]
    - Mask ratio from uniform distribution [min_mask_ratio, max_mask_ratio]

    Ensures at least 1 block remains unmasked.

    Args:
        x: Tensor of shape (seq_len,) or (batch, seq_len)
        min_block_size: Minimum block size (default: 1)
        max_block_size: Maximum block size (default: seq_len // 2)
        min_mask_ratio: Minimum masking fraction (default: 0.1)
        max_mask_ratio: Maximum masking fraction (default: 0.9)

    Returns:
        masked_x: Input with masked blocks set to 0
        mask: Boolean tensor, True where input is masked
        block_size: The sampled block size (for logging)
        mask_ratio: The sampled mask ratio (for logging)
    """
    # Determine sequence length
    if x.dim() == 1:
        seq_len = x.shape[0]
        batch_mode = False
    else:
        seq_len = x.shape[-1]
        batch_mode = True

    # Set default max block size to half the sequence length
    if max_block_size is None:
        max_block_size = seq_len // 2

    # Ensure max_block_size is valid
    max_block_size = min(max_block_size, seq_len)

    # Sample random block size and mask ratio
    block_size = np.random.randint(min_block_size, max_block_size + 1)
    mask_ratio = np.random.uniform(min_mask_ratio, max_mask_ratio)

    # Calculate number of blocks
    num_blocks = int(np.ceil(seq_len / block_size))
    num_masked = int(num_blocks * mask_ratio)

    # Ensure at least 1 block is unmasked
    num_masked = min(num_masked, num_blocks - 1)
    num_masked = max(num_masked, 1)  # Ensure at least 1 block is masked

    # Randomly select blocks to mask
    masked_blocks = np.random.choice(np.arange(num_blocks), size=num_masked, replace=False)

    # Create masked version
    x_masked = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool)

    for block_idx in masked_blocks:
        start = block_idx * block_size
        end = min(start + block_size, seq_len)

        if batch_mode:
            x_masked[..., start:end] = 0.0
            mask[..., start:end] = True
        else:
            x_masked[start:end] = 0.0
            mask[start:end] = True

    return x_masked, mask, block_size, mask_ratio
