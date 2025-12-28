import argparse
from pathlib import Path
import h5py
from lcgen.utils.mask import apply_block_mask
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
from pathlib import Path as _P
# Ensure local 'src' is on sys.path so we can import lcgen
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'src'))
from lcgen.models.simple_min_gru import SimpleMinGRU
from lcgen.utils.trunc_data import extract_data
from lcgen.utils.loss import recon_loss

class TimeSeriesDataset(Dataset):
    def __init__(self, h5path, random_seed, min_size, max_size, mask_portion, max_length=16384, num_samples=None, mock_sinusoid=False, mock_noise=0.1):
        self.min_size = min_size
        self.max_size = max_size
        self.mask_portion = mask_portion
        self.rng = np.random.default_rng(random_seed)
        self.mock_noise = mock_noise
        
        p = Path(h5path)
        if p.exists():
            if max_length < 16384:
                data_dict = extract_data(h5path, random_seed=random_seed,keys=['flux','flux_err','time'], max_length=max_length, num_samples=num_samples)
                self.flux = data_dict['flux']
                self.time = data_dict['time']
                self.flux_err = data_dict['flux_err']
            else:
                with h5py.File(h5path, 'r') as f:
                    self.flux = f['flux'][:]
                    self.time = f['time'][:]
                    self.flux_err = f['flux_err'][:]
            self.flux = np.asarray(self.flux, dtype=np.float32)
            self.time = np.asarray(self.time, dtype=np.float32)
            self.flux_err = np.asarray(self.flux_err, dtype=np.float32)
        else:
            # Fall back to a synthetic dataset if file is missing (small and fast).
            print(f'Warning: {h5path} not found. Using synthetic data for smoke test.')
            N = 128
            L = 256
            self.flux = np.random.randn(N, L).astype(np.float32)
            self.flux_err = np.random.randn(N, L).astype(np.float32)
            self.time = np.random.randn(N, L).astype(np.float32)
        if mock_sinusoid:
            # Generate a mock sinusoidal dataset
            for i in range(len(self.flux)):
                t_range_i = np.max(self.time[i]) - np.min(self.time[i])
                phase = np.random.uniform(0, 2 * np.pi)
                amp = np.random.normal(1, 0.5)  # Random amplitude
                num_cycles = np.abs(np.random.normal(5, 5))  # Random number of cycles
                period = t_range_i / num_cycles
                t_reg_space = np.linspace(np.min(self.time[i]), np.max(self.time[i]), self.flux.shape[1])
                flux_reg_space = amp * np.sin(t_reg_space * (2 * np.pi / period) + phase)
                flux_at_t = np.interp(self.time[i], t_reg_space, flux_reg_space)
                flux_at_t_noisy = flux_at_t + np.random.normal(0, self.mock_noise, size=flux_at_t.shape)  # Add noise
                self.flux[i] = flux_at_t_noisy
                self.flux_err[i] = np.abs(np.random.normal(1,0.5, size=self.flux.shape[1]))+0.05

    def __len__(self):
        return len(self.flux)

    def _generate_block_mask(self, L):
        """Generate a random block mask for a single sample.
        
        Args:
            L: sequence length
            
        Returns:
            mask: np.ndarray of shape (L,) with 1=observed, 0=masked
        """
        if self.mask_portion <= 0:
            return np.ones(L, dtype=np.float32)
        
        mask = np.ones(L, dtype=np.float32)
        total_mask_size = int(L * self.mask_portion)
        
        if total_mask_size == 0:
            return mask
            
        masked_so_far = 0
        while masked_so_far < total_mask_size:
            block_size = self.rng.integers(self.min_size, self.max_size + 1)
            if masked_so_far + block_size > total_mask_size:
                block_size = total_mask_size - masked_so_far
            if block_size <= 0:
                break
            max_start = L - block_size
            if max_start < 0:
                break
            start_idx = self.rng.integers(0, max_start + 1)
            mask[start_idx:start_idx + block_size] = 0.0
            masked_so_far += block_size
        
        return mask

    def __getitem__(self, idx):
        flux = self.flux[idx]
        flux_err = self.flux_err[idx]
        time = self.time[idx]
        L = len(flux)
        
        # Generate a fresh random mask for this sample
        mask = self._generate_block_mask(L)
        
        return [flux, flux_err, time, mask]
    
    def apply_block_mask(self, min_size, max_size, mask_portion):
        """
        Apply random block masks to 2D or 3D time series data (B, L) or (B, L, C).
        Each sample has its own random masked blocks; all channels in a sample share the same mask.

        Args:
            data: np.ndarray, shape (B, L) or (B, L, C)
            min_size: minimum block size
            max_size: maximum block size
            mask_portion: fraction of time steps to mask (0-1)

        Returns:
            masked_flux: same shape as flux
            masked_flux_err: same shape as flux_err
            mask: same shape as flux (1=unmasked, 0=masked)
        """
        masked_flux = self.flux.copy()
        masked_flux_err = self.flux_err.copy()
        mask = np.ones_like(self.flux, dtype=np.float32)
        
        B, L = self.flux.shape[:2]
        C = self.flux.shape[2] if self.flux.ndim == 3 else 1
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
            if self.flux.ndim == 2:
                masked_flux[i, sample_mask] = 0.0
                masked_flux_err[i, sample_mask] = 0.0
                mask[i, sample_mask] = 0.0
            else:  # (B, L, C)
                masked_flux[i, sample_mask, :] = 0.0
                masked_flux_err[i, sample_mask, :] = 0.0
                mask[i, sample_mask, :] = 0.0
        
        print(f'Applied block mask: min_size={min_size}, max_size={max_size}, mask_portion={mask_portion}')

        self.masked_flux = masked_flux
        self.masked_flux_err = masked_flux_err
        self.mask = mask

def collate_fn(batch):
    """Collate list of samples into batched tensors.

    Each item in batch is [flux, flux_err, time, mask] where each is a 1D numpy
    array of length L. Return a tuple of torch.FloatTensors: (flux, flux_err, time, mask)
    with shapes (B, L).
    """
    fluxes = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
    flux_errs = np.stack([b[1] for b in batch], axis=0).astype(np.float32)
    times = np.stack([b[2] for b in batch], axis=0).astype(np.float32)
    masks = np.stack([b[3] for b in batch], axis=0).astype(np.float32)

    return (torch.from_numpy(fluxes), torch.from_numpy(flux_errs), torch.from_numpy(times), torch.from_numpy(masks))