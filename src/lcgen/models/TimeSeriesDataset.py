import argparse
from pathlib import Path
import h5py
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
    def __init__(self, h5path, random_seed,max_length=16384, num_samples=None):
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
            self.masked_flux = self.flux.copy()
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
        self.apply_mask()

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        return [self.flux[idx], self.flux_err[idx], self.time[idx],
                self.mask[idx], self.masked_flux[idx]]
    
    def apply_mask(self):
        self.mask = np.full(self.flux.shape, 1.0, dtype=np.float32)
        self.masked_flux = self.flux * self.mask


def collate_fn(batch):
    """Collate list of samples into batched tensors.

    Each item in batch is [flux, flux_err, time] where each is a 1D numpy
    array of length L. Return a tuple of torch.FloatTensors: (flux, flux_err, time)
    with shapes (B, L).
    """
    fluxes = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
    flux_errs = np.stack([b[1] for b in batch], axis=0).astype(np.float32)
    times = np.stack([b[2] for b in batch], axis=0).astype(np.float32)
    masks = np.stack([b[3] for b in batch], axis=0).astype(np.float32)
    masked_fluxes = np.stack([b[4] for b in batch], axis=0).astype(np.float32)
    return (torch.from_numpy(masked_fluxes), torch.from_numpy(flux_errs), torch.from_numpy(times), torch.from_numpy(masks), torch.from_numpy(masked_fluxes))