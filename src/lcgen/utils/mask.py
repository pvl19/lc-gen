import numpy as np

def apply_block_mask(data, min_size, max_size, mask_portion):
    """
    Apply random block masks to 2D or 3D time series data (B, L) or (B, L, C).
    Each sample has its own random masked blocks; all channels in a sample share the same mask.

    Args:
        data: np.ndarray, shape (B, L) or (B, L, C)
        min_size: minimum block size
        max_size: maximum block size
        mask_portion: fraction of time steps to mask (0-1)

    Returns:
        masked_data: same shape as data
        mask: same shape as data (1=unmasked, 0=masked)
    """
    masked_data = data.copy()
    mask = np.ones_like(data, dtype=np.float32)
    
    B, L = data.shape[:2]
    C = data.shape[2] if data.ndim == 3 else 1
    total_mask_size = int(L * mask_portion)

    # For each sample, generate a mask
    for i in range(B):
        # Create a boolean array of length L: 1 = masked, 0 = unmasked
        sample_mask = np.zeros(L, dtype=np.bool_)

        # Precompute blocks that fit into total_mask_size
        masked_so_far = 0
        while masked_so_far < total_mask_size:
            block_size = np.random.randint(min_size, max_size + 1)
            if masked_so_far + block_size > total_mask_size:
                block_size = total_mask_size - masked_so_far
            start_idx = np.random.randint(0, L - block_size + 1)
            sample_mask[start_idx:start_idx + block_size] = True
            masked_so_far += block_size

        # Apply mask to data
        if data.ndim == 2:
            masked_data[i, sample_mask] = 0.0
            mask[i, sample_mask] = 0.0
        else:  # (B, L, C)
            masked_data[i, sample_mask, :] = 0.0
            mask[i, sample_mask, :] = 0.0

    return masked_data, mask
