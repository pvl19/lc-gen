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

from lcgen.models.MetadataAgePredictor import DEFAULT_METADATA_FIELDS, MetadataStandardizer

# Default metadata feature names — mirrors DEFAULT_METADATA_FIELDS from MetadataAgePredictor
METADATA_FEATURES = DEFAULT_METADATA_FIELDS


class LazyH5Dataset(Dataset):
    """Memory-efficient dataset that reads flux/flux_err/time from HDF5 on demand.

    Metadata (small 1D arrays) is loaded eagerly at init. Flux arrays are read
    per-sample from disk, making this suitable for large datasets and multi-GPU
    DDP training where loading all data into RAM per process would be infeasible.

    Designed to work with DataLoader num_workers > 0: each worker process opens
    its own h5 file handle on first access.
    """

    def __init__(self, h5path, random_seed, min_size, max_size, mask_portion,
                 num_samples=None, use_metadata=False, metadata_features=None):
        self.h5path = str(h5path)
        self.min_size = min_size
        self.max_size = max_size
        self.mask_portion = mask_portion
        self.rng = np.random.default_rng(random_seed)
        self.use_metadata = use_metadata
        self.metadata_features = metadata_features or METADATA_FEATURES
        self.num_meta_features = len(self.metadata_features)
        self._h5 = None  # opened lazily per worker process

        with h5py.File(self.h5path, 'r') as f:
            n_total = int(f.attrs['n_samples'])
            lengths = f['length'][:].astype(np.int32)

            if use_metadata:
                meta_grp = f['metadata']
                missing = [fld for fld in self.metadata_features if fld not in meta_grp]
                if missing:
                    raise RuntimeError(f'Missing metadata features in {h5path}: {missing}')
                raw = {fld: meta_grp[fld][:].astype(np.float32) for fld in self.metadata_features}
                standardizer = MetadataStandardizer(fields=self.metadata_features)
                metadata = standardizer.transform(raw)
            else:
                metadata = None

        # Subsample if requested
        if num_samples is not None and num_samples < n_total:
            rng = np.random.default_rng(random_seed)
            idx = rng.choice(n_total, num_samples, replace=False)
            idx.sort()
        else:
            idx = np.arange(n_total)

        self.indices = idx
        self.lengths = lengths[idx]
        self.metadata = metadata[idx] if metadata is not None else None

    def _get_h5(self):
        """Return an open h5py file handle, opening lazily on first access per worker."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5path, 'r')
        return self._h5

    def __len__(self):
        return len(self.indices)

    def _generate_block_mask(self, L):
        if self.mask_portion <= 0:
            return np.ones(L, dtype=np.float32)
        mask = np.ones(L, dtype=np.float32)
        total_mask_size = int(L * self.mask_portion)
        if total_mask_size == 0:
            return mask
        masked_so_far = 0
        while masked_so_far < total_mask_size:
            log_size = self.rng.uniform(np.log10(self.min_size), np.log10(self.max_size))
            block_size = int(round(10 ** log_size))
            block_size = max(self.min_size, min(block_size, self.max_size))
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
        h5_idx = int(self.indices[idx])
        L = int(self.lengths[idx])
        h5 = self._get_h5()
        flux     = h5['flux'][h5_idx, :L].astype(np.float32)
        flux_err = h5['flux_err'][h5_idx, :L].astype(np.float32)
        time     = h5['time'][h5_idx, :L].astype(np.float32)
        mask = self._generate_block_mask(L)
        meta = self.metadata[idx] if (self.use_metadata and self.metadata is not None) \
               else np.zeros(self.num_meta_features, dtype=np.float32)
        empty = np.zeros(0, dtype=np.float32)
        return [flux, flux_err, time, mask, L, meta, empty, empty, empty]


class TimeSeriesDataset(Dataset):
    def __init__(self, h5path, random_seed, min_size, max_size, mask_portion, max_length=16384, num_samples=None, mock_sinusoid=False, mock_noise=0.1, use_metadata=False, metadata_features=None, use_conv_channels=False):
        self.min_size = min_size
        self.max_size = max_size
        self.mask_portion = mask_portion
        self.rng = np.random.default_rng(random_seed)
        self.mock_noise = mock_noise
        self.use_metadata = use_metadata
        self.use_conv_channels = use_conv_channels
        self.metadata_features = metadata_features or METADATA_FEATURES
        self.num_meta_features = len(self.metadata_features)
        
        p = Path(h5path)
        if p.exists():
            if max_length < 16384:
                data_dict = extract_data(h5path, random_seed=random_seed,keys=['flux','flux_err','time'], max_length=max_length, num_samples=num_samples)
                self.flux = data_dict['flux']
                self.time = data_dict['time']
                self.flux_err = data_dict['flux_err']
                self.lengths = data_dict['length']  # Actual sequence lengths (capped at max_length)
                # Get the indices used by extract_data so we can load matching metadata
                sample_indices = data_dict.get('indices')
                # Load metadata directly from H5 (it's a group, not a dataset)
                if use_metadata:
                    with h5py.File(h5path, 'r') as f:
                        if 'metadata' in f:
                            meta_grp = f['metadata']
                            # Load all metadata first, then subset
                            full_metadata = self._load_metadata_from_h5(meta_grp, f['flux'].shape[0])
                            if sample_indices is not None:
                                self.metadata = full_metadata[sample_indices]
                            else:
                                self.metadata = full_metadata[:len(self.flux)]
                        else:
                            self.metadata = None
                else:
                    self.metadata = None

                # Load convolutional channel data (power spectra, f-stat, ACF)
                if use_conv_channels:
                    self.power, self.f_stat, self.acf = self._load_conv_channels(
                        h5path, sample_indices=sample_indices, n_flux=len(self.flux))
                else:
                    self.power = None
                    self.f_stat = None
                    self.acf = None
            else:
                with h5py.File(h5path, 'r') as f:
                    self.flux = f['flux'][:]
                    self.time = f['time'][:]
                    self.flux_err = f['flux_err'][:]
                    # Load actual lengths if available
                    if 'length' in f:
                        self.lengths = f['length'][:]
                    else:
                        self.lengths = np.full(len(self.flux), self.flux.shape[1], dtype=np.int32)
                    # Load stellar metadata if available and requested
                    if use_metadata and 'metadata' in f:
                        meta_grp = f['metadata']
                        self.metadata = self._load_metadata_from_h5(meta_grp, len(self.flux))
                    else:
                        self.metadata = None
                    # Load convolutional channel data
                    if use_conv_channels:
                        self.power, self.f_stat, self.acf = self._load_conv_channels(
                            h5path, sample_indices=None, n_flux=len(self.flux))
                    else:
                        self.power = None
                        self.f_stat = None
                        self.acf = None
            self.flux = np.asarray(self.flux, dtype=np.float32)
            self.time = np.asarray(self.time, dtype=np.float32)
            self.flux_err = np.asarray(self.flux_err, dtype=np.float32)
            self.lengths = np.asarray(self.lengths, dtype=np.int32)
            if use_conv_channels and self.power is not None:
                self.power = np.asarray(self.power, dtype=np.float32)
                self.f_stat = np.asarray(self.f_stat, dtype=np.float32)
                self.acf = np.asarray(self.acf, dtype=np.float32)
        else:
            # Fall back to a synthetic dataset if file is missing (small and fast).
            print(f'Warning: {h5path} not found. Using synthetic data for smoke test.')
            N = 128
            L = 256
            self.flux = np.random.randn(N, L).astype(np.float32)
            self.flux_err = np.random.randn(N, L).astype(np.float32)
            self.time = np.random.randn(N, L).astype(np.float32)
            self.lengths = np.full(N, L, dtype=np.int32)  # All positions valid for synthetic
            self.metadata = None
            # Synthetic conv data
            if use_conv_channels:
                self.power = np.random.randn(N, 2500).astype(np.float32)
                self.f_stat = np.random.randn(N, 2500).astype(np.float32)
                self.acf = np.random.randn(N, 2500).astype(np.float32)
            else:
                self.power = None
                self.f_stat = None
                self.acf = None
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

    @staticmethod
    def _load_conv_channels(h5path, sample_indices=None, n_flux=None):
        """Load power/f_stat/acf from a sidecar *_spectra.h5 or from the main H5.

        Lookup order:
          1. <stem>_spectra.h5  (separate file, preferred — keeps originals safe)
          2. Datasets inside h5path itself (legacy / in-file storage)
        """
        # Try sidecar file first
        stem = h5path.rsplit('.h5', 1)[0]
        spectra_path = stem + '_spectra.h5'
        conv_path = spectra_path if Path(spectra_path).exists() else h5path

        with h5py.File(conv_path, 'r') as f:
            if 'power' in f and 'f_stat' in f and 'acf' in f:
                if sample_indices is not None:
                    power = f['power'][:][sample_indices]
                    f_stat = f['f_stat'][:][sample_indices]
                    acf = f['acf'][:][sample_indices]
                else:
                    power = f['power'][:n_flux] if n_flux else f['power'][:]
                    f_stat = f['f_stat'][:n_flux] if n_flux else f['f_stat'][:]
                    acf = f['acf'][:n_flux] if n_flux else f['acf'][:]
                return power, f_stat, acf

        print(f"Warning: Conv channel data (power/f_stat/acf) not found in "
              f"{conv_path} or {h5path}")
        return None, None, None

    def _load_metadata_from_h5(self, meta_grp, n_samples):
        """Load and normalise metadata features from H5 group using MetadataStandardizer."""
        missing = [f for f in self.metadata_features if f not in meta_grp]
        if missing:
            raise RuntimeError(f"Missing required metadata features in H5 file: {missing}")

        data = {}
        for feat_name in self.metadata_features:
            vals = meta_grp[feat_name][:]
            if vals.dtype.kind == 'S':
                vals = vals.astype(float)
            data[feat_name] = vals.astype(np.float32)

        standardizer = MetadataStandardizer(fields=self.metadata_features)
        return standardizer.transform(data)  # (n_samples, num_features), NaNs → 0

    def __len__(self):
        return len(self.flux)

    def _generate_block_mask(self, L):
        """Generate a random block mask for a single sample.

        Block sizes are sampled from a log-uniform distribution between min_size and max_size,
        which gives more smaller blocks and fewer larger blocks.

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
            # Sample block size from log-uniform distribution
            # This favors smaller blocks (more small blocks, fewer large blocks)
            log_min = np.log10(self.min_size)
            log_max = np.log10(self.max_size)
            log_size = self.rng.uniform(log_min, log_max)
            block_size = int(round(10 ** log_size))
            # Clamp to valid range [min_size, max_size]
            block_size = max(self.min_size, min(block_size, self.max_size))

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
        actual_length = int(self.lengths[idx])

        # Truncate to actual length — padding is handled dynamically in collate_fn
        flux = self.flux[idx][:actual_length]
        flux_err = self.flux_err[idx][:actual_length]
        time = self.time[idx][:actual_length]

        # Generate block mask over actual data only (all positions real, no padding mask needed)
        mask = self._generate_block_mask(actual_length)

        # Get metadata if available
        if self.use_metadata and self.metadata is not None:
            meta = self.metadata[idx]
        else:
            meta = np.zeros(self.num_meta_features, dtype=np.float32)

        # Get convolutional channel data if available
        if self.use_conv_channels and self.power is not None:
            power = self.power[idx]
            f_stat = self.f_stat[idx]
            acf = self.acf[idx]
        else:
            # Return empty arrays if not using conv channels
            power = np.zeros(0, dtype=np.float32)
            f_stat = np.zeros(0, dtype=np.float32)
            acf = np.zeros(0, dtype=np.float32)

        return [flux, flux_err, time, mask, actual_length, meta, power, f_stat, acf]
    
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
    """Collate list of samples into batched tensors with dynamic padding.

    Each item in batch is [flux, flux_err, time, mask, actual_length, metadata, power, f_stat, acf]
    where flux/flux_err/time/mask are 1D numpy arrays of varying length (actual_length),
    metadata is a 1D array of stellar features,
    and power/f_stat/acf are 1D arrays of frequency-domain features.

    Sequences are padded with zeros to the longest sequence in the batch.
    Padded positions have mask=0, so they are excluded from the loss automatically.

    Return a tuple of torch.FloatTensors: (flux, flux_err, time, mask, metadata, conv_data)
    with shapes (B, L_max) for time series, (B, num_meta_features) for metadata,
    and conv_data is a dict with power/f_stat/acf of shape (B, freq_length) or None.
    """
    lengths = np.array([b[4] for b in batch], dtype=np.int32)
    batch_max_len = int(lengths.max())

    def pad(arr):
        out = np.zeros(batch_max_len, dtype=np.float32)
        out[:len(arr)] = arr
        return out

    fluxes    = np.stack([pad(b[0]) for b in batch])
    flux_errs = np.stack([pad(b[1]) for b in batch])
    times     = np.stack([pad(b[2]) for b in batch])
    masks     = np.stack([pad(b[3]) for b in batch])  # block mask, zero-padded → padding positions get mask=0
    metadata  = np.stack([b[5] for b in batch]).astype(np.float32)

    # Check if conv data is present (non-empty arrays)
    if len(batch[0][6]) > 0:
        power  = np.stack([b[6] for b in batch]).astype(np.float32)
        f_stat = np.stack([b[7] for b in batch]).astype(np.float32)
        acf    = np.stack([b[8] for b in batch]).astype(np.float32)
        conv_data = {
            'power': torch.from_numpy(power),
            'f_stat': torch.from_numpy(f_stat),
            'acf': torch.from_numpy(acf)
        }
    else:
        conv_data = None

    return (torch.from_numpy(fluxes), torch.from_numpy(flux_errs),
            torch.from_numpy(times), torch.from_numpy(masks),
            torch.from_numpy(metadata), conv_data)