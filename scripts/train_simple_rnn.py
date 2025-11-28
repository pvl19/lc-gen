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


class TimeSeriesDataset(Dataset):
    def __init__(self, h5path):
        p = Path(h5path)
        if p.exists():
            with h5py.File(h5path, 'r') as f:
                self.flux = f['flux'][:]
            self.flux = np.asarray(self.flux, dtype=np.float32)
        else:
            # Fall back to a synthetic dataset if file is missing (small and fast).
            print(f'Warning: {h5path} not found. Using synthetic data for smoke test.')
            N = 128
            L = 256
            self.flux = np.random.randn(N, L).astype(np.float32)

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        return self.flux[idx]


def collate_fn(batch):
    arr = np.stack(batch, axis=0)  # (B, L)
    return torch.from_numpy(arr)


def train(args):
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    ds = TimeSeriesDataset(args.input)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = SimpleMinGRU(hidden_size=args.hidden_size, use_flow=args.use_flow).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    if args.epochs <= 0:
        print('No epochs requested, exiting.')
        return

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            x = batch.to(device)
            out = model(x)
            recon = out['reconstructed']  # (B, L, 2)
            mean = recon[..., 0]
            logs = recon[..., 1]

            # Simple Gaussian NLL on mean+sigma (sigma = softplus(logs))
            sigma = F.softplus(logs) + 1e-6
            var = sigma ** 2
            nll = 0.5 * (torch.log(2 * math.pi * var) + (mean - x) ** 2 / var)
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
    p.add_argument('--input', type=str, default='data/real_lightcurves/timeseries_2048_25pct.h5')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--output_dir', type=str, default='tmp/simple_rnn')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--use_flow', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
