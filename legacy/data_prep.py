import numpy as np
import torch

def normalize_ps(data, scale_factor=50):
    asinh = np.arcsinh(np.log10(data) / scale_factor)
    scaled = (asinh - np.mean(asinh)) / np.std(asinh)
    return scaled

def normalize_acf(data, scale_factor=0.001):
    asinh = np.arcsinh(data / scale_factor)
    scaled = (asinh - np.mean(asinh)) / np.std(asinh)
    return scaled

def block_predefined_mask(x,block_size=10,mask_ratio=0.7):
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
    return x,mask

def block_predefined_mask_lc(x,block_size=10,mask_ratio=0.7):
    # Accepts x of shape (seq_len, 2) or (batch, seq_len, 2)
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
            x_masked[i, start:end, 0] = 0.0
            x_masked[i, start:end, 1] = 16 * x_masked[i, start:end, 1]
            mask[i, start:end] = True

    if single_input:
        return x_masked[0], mask[0]
    else:
        return x_masked, mask

def block_mask(x, block_size=100, num_blocks=1):
    """
    Mask contiguous blocks by setting them to 0.

    Args:
        x: Tensor of shape (seq_len,) or (channels, seq_len)
        block_size: length of each masked block
        num_blocks: how many blocks to mask

    Returns:
        masked_x: input with masked blocks
        mask: boolean tensor, True where input is kept, False where masked
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
