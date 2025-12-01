"""Minimal training script for the single-cell SimpleMinGRU model.

This script is intentionally small: it loads an HDF5 timeseries file, creates
a simple dataloader, trains the single-cell model for a few epochs, and saves
a checkpoint. It supports the optional zuko flow if available; otherwise it
falls back to Gaussian negative-log-likelihood (or MSE for simplicity).
"""
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

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        return [self.flux[idx], self.flux_err[idx], self.time[idx]]


def collate_fn(batch):
    """Collate list of samples into batched tensors.

    Each item in batch is [flux, flux_err, time] where each is a 1D numpy
    array of length L. Return a tuple of torch.FloatTensors: (flux, flux_err, time)
    with shapes (B, L).
    """
    fluxes = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
    flux_errs = np.stack([b[1] for b in batch], axis=0).astype(np.float32)
    times = np.stack([b[2] for b in batch], axis=0).astype(np.float32)
    return (torch.from_numpy(fluxes), torch.from_numpy(flux_errs), torch.from_numpy(times))


def train(args):
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    ds = TimeSeriesDataset(args.input, args.random_seed, args.max_length, args.num_samples)
    print('Shape of TimeSeriesDataset:', ds.flux.shape)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = SimpleMinGRU(hidden_size=args.hidden_size, use_flow=args.use_flow).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.epochs <= 0:
        print('No epochs requested, exiting.')
        return

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            # batch is a tuple: (flux, flux_err, time) each shaped (B, L)
            flux, flux_err, times = batch
            # print('mean flux:', flux.mean().item(), 'std flux:', flux.std().item())
            # print('max flux_err:', flux_err.max().item(), 'min flux_err:', flux_err.min().item())
            flux = flux.to(device)
            flux_err = flux_err.to(device)

            # Build input channels [flux, flux_err] -> (B, L, 2)
            x_in = torch.stack([flux, flux_err], dim=-1)

            out = model(x_in)
            recon = out['reconstructed']  # (B, L, 1) -> [mean, raw_logsigma]
            mean = recon[..., 0]

            nll = recon_loss(flux, flux_err, mean)
            loss = nll.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        avg = total_loss / max(1, n)
        print(f'Epoch {epoch+1}/{args.epochs} - Loss: {avg:.6f}')

    # Save final model
    outp = Path(args.output_dir)
    outp.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, outp / 'simple_min_gru_final.pt')
    print('Saved final model to', outp / 'simple_min_gru_final.pt')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='data/timeseries.h5')
    p.add_argument('--max_length', type=int, default=16384)
    p.add_argument('--random_seed', type=int, default=19)
    p.add_argument('--num_samples', type=int, default=None)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--output_dir', type=str, default='output/simple_rnn')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--use_flow', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
