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


# Global cache for valid masking combinations
_MASKING_COMBINATIONS_CACHE = {}


def _compute_valid_combinations(seq_len, min_block_size, max_block_size,
                                min_mask_ratio, max_mask_ratio):
    """
    Precompute valid (block_size, num_masked_blocks, actual_ratio) combinations.

    This is cached to avoid recomputing on every call.
    """
    cache_key = (seq_len, min_block_size, max_block_size, min_mask_ratio, max_mask_ratio)

    if cache_key in _MASKING_COMBINATIONS_CACHE:
        return _MASKING_COMBINATIONS_CACHE[cache_key]

    # Generate all powers of 2 within the valid range
    min_power = int(np.floor(np.log2(max(1, min_block_size))))
    max_power = int(np.floor(np.log2(max_block_size)))

    if max_power < min_power:
        max_power = min_power

    valid_block_sizes = [2**p for p in range(min_power, max_power + 1)]

    # For each block size, find valid numbers of masked blocks
    valid_combinations = []

    for block_size in valid_block_sizes:
        num_blocks = int(np.ceil(seq_len / block_size))

        # Precompute total data in all blocks
        total_data_in_blocks = sum([min((i+1)*block_size, seq_len) - i*block_size
                                   for i in range(num_blocks)])
        avg_block_size = total_data_in_blocks / num_blocks

        # Try each possible number of masked blocks
        for num_masked in range(1, num_blocks):  # At least 1 unmasked
            expected_masked_data = num_masked * avg_block_size
            actual_ratio = expected_masked_data / seq_len

            # Check if this combination gives us the desired masking percentage
            if min_mask_ratio <= actual_ratio <= max_mask_ratio:
                valid_combinations.append((block_size, num_masked, actual_ratio))

    # Cache the result
    _MASKING_COMBINATIONS_CACHE[cache_key] = valid_combinations
    return valid_combinations


def dynamic_block_mask(x, min_block_size=1, max_block_size=None,
                       min_mask_ratio=0.1, max_mask_ratio=0.9):
    """
    Create dynamic block mask with power-of-2 block sizes (FAST version).

    Randomly samples:
    - Block size from powers of 2 (log-style distribution)
    - Mask ratio uniformly from [min_mask_ratio, max_mask_ratio]

    Much faster than validated version - skips expensive combination checking.

    Args:
        x: Tensor of shape (seq_len,) or (batch, seq_len)
        min_block_size: Minimum block size (default: 1)
        max_block_size: Maximum block size (default: seq_len // 2)
        min_mask_ratio: Minimum fraction of blocks to mask (default: 0.1)
        max_mask_ratio: Maximum fraction of blocks to mask (default: 0.9)

    Returns:
        masked_x: Input with masked blocks set to 0
        mask: Boolean tensor, True where input is masked
        block_size: The sampled block size (for logging)
        actual_mask_ratio: The actual fraction of data masked (for logging)
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

    # Sample block size from powers of 2 (log-style distribution) - FAST!
    min_power = int(np.floor(np.log2(max(1, min_block_size))))
    max_power = int(np.floor(np.log2(max_block_size)))
    if max_power < min_power:
        max_power = min_power

    # Randomly select a power of 2
    power = np.random.randint(min_power, max_power + 1)
    block_size = 2 ** power

    # Sample mask ratio uniformly
    mask_ratio = np.random.uniform(min_mask_ratio, max_mask_ratio)

    # Calculate number of blocks
    num_blocks = int(np.ceil(seq_len / block_size))
    num_masked = int(num_blocks * mask_ratio)

    # Ensure at least 1 block is unmasked and at least 1 is masked
    num_masked = max(1, min(num_masked, num_blocks - 1))

    # Randomly select blocks to mask
    masked_blocks = np.random.choice(num_blocks, size=num_masked, replace=False)

    # Create boolean mask efficiently using vectorized operations
    mask = torch.zeros(seq_len, dtype=torch.bool)

    # Vectorize the masking by computing all indices at once
    for block_idx in masked_blocks:
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        mask[start:end] = True

    # Calculate actual mask ratio
    actual_mask_ratio = mask.sum().item() / seq_len

    # Apply mask to data
    x_masked = x.clone()
    if batch_mode:
        x_masked[..., mask] = 0.0
    else:
        x_masked[mask] = 0.0

    return x_masked, mask, block_size, actual_mask_ratio


def dynamic_block_mask_batch(x_batch, min_block_size=1, max_block_size=None,
                             min_mask_ratio=0.1, max_mask_ratio=0.9):
    """
    Create dynamic block masks for an entire batch at once (MUCH faster).

    Picks ONE (block_size, mask_ratio) combination for the batch, but varies
    which blocks are masked per sample.

    Args:
        x_batch: Tensor of shape (batch_size, seq_len)
        min_block_size: Minimum block size (default: 1)
        max_block_size: Maximum block size (default: seq_len // 2)
        min_mask_ratio: Minimum fraction of DATA to mask (default: 0.1)
        max_mask_ratio: Maximum fraction of DATA to mask (default: 0.9)

    Returns:
        masked_batch: Batch with masked blocks set to 0
        mask_batch: Boolean tensor [batch_size, seq_len], True where masked
        block_size: The sampled block size (same for all samples)
        actual_ratio: The actual fraction of data masked (same for all samples)
    """
    batch_size, seq_len = x_batch.shape

    # Set default max block size
    if max_block_size is None:
        max_block_size = seq_len // 2

    max_block_size = min(max_block_size, seq_len)

    # Get valid combinations (cached) - ONLY ONCE for the batch
    valid_combinations = _compute_valid_combinations(
        seq_len, min_block_size, max_block_size, min_mask_ratio, max_mask_ratio
    )

    # Pick one combination for the entire batch
    if len(valid_combinations) == 0:
        # Fallback
        min_power = int(np.floor(np.log2(max(1, min_block_size))))
        max_power = int(np.floor(np.log2(max_block_size)))
        if max_power < min_power:
            max_power = min_power
        valid_block_sizes = [2**p for p in range(min_power, max_power + 1)]

        block_size = valid_block_sizes[len(valid_block_sizes) // 2]
        num_blocks = int(np.ceil(seq_len / block_size))
        num_masked = max(1, min(num_blocks - 1, int(num_blocks * min_mask_ratio)))
        actual_ratio = (num_masked * block_size) / seq_len
    else:
        block_size, num_masked, actual_ratio = valid_combinations[
            np.random.randint(len(valid_combinations))
        ]
        num_blocks = int(np.ceil(seq_len / block_size))

    # VECTORIZED: Generate all random block selections at once
    # Shape: [batch_size, num_masked] - which blocks to mask for each sample
    masked_blocks_batch = np.array([
        np.random.choice(num_blocks, size=num_masked, replace=False)
        for _ in range(batch_size)
    ])  # [batch_size, num_masked]

    # Create mask tensor
    mask_batch = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # VECTORIZED: Set all masked regions at once
    # For each masked block position
    for i in range(num_masked):
        # Get the block index for this position across all samples
        block_indices = masked_blocks_batch[:, i]  # [batch_size]

        # Calculate start/end for each sample
        starts = block_indices * block_size
        ends = np.minimum(starts + block_size, seq_len)

        # Apply mask for all samples
        for j in range(batch_size):
            mask_batch[j, starts[j]:ends[j]] = True

    # Apply masking: set masked regions to 0 (vectorized operation)
    x_masked_batch = x_batch.clone()
    x_masked_batch[mask_batch] = 0.0

    return x_masked_batch, mask_batch, block_size, actual_ratio
