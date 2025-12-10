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
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader

import sys
from pathlib import Path as _P
# Ensure local 'src' is on sys.path so we can import lcgen
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from lcgen.models.simple_min_gru import SimpleMinGRU, BiDirectionalMinGRU
from lcgen.utils.trunc_data import extract_data
from lcgen.utils.loss import recon_loss
from lcgen.models.TimeSeriesDataset import TimeSeriesDataset, collate_fn

def train(args):
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    ds = TimeSeriesDataset(args.input, args.random_seed,  args.min_size, args.max_size, args.mask_portion, args.max_length, args.num_samples, args.mock_sinusoid)
    print('Shape of TimeSeriesDataset:', ds.flux.shape)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    if args.direction == 'bi':
        model = BiDirectionalMinGRU(hidden_size=args.hidden_size, use_flow=args.use_flow).to(device)
    else:
        model = SimpleMinGRU(hidden_size=args.hidden_size, direction=args.direction, use_flow=args.use_flow).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    # scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(loader), epochs=args.epochs, pct_start=0.3, div_factor=10.0, final_div_factor=1000.0)

    if args.epochs <= 0:
        print('No epochs requested, exiting.')
        return

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            optimizer.zero_grad()
            # batch is a tuple: (flux, flux_err, time) each shaped (B, L)
            flux, flux_err, times, mask = batch
            # print('mean flux:', flux.mean().item(), 'std flux:', flux.std().item())
            # print('max flux_err:', flux_err.max().item(), 'min flux_err:', flux_err.min().item())
            flux = flux.to(device)
            flux_err = flux_err.to(device)
            times = times.to(device)
            # masked_flux = masked_flux.to(device)
            # masked_flux_err = masked_flux_err.to(device)
            mask = mask.to(device)

            # Build input channels [flux, flux_err] -> (B, L, 2)
            x_in = torch.stack([flux, flux_err, mask], dim=-1)
            t_in = torch.stack([times], dim=-1)
            out = model(x_in, t_in)
            recon = out['reconstructed']  # (B, L, 1) -> [mean, raw_logsigma]
            mean = recon[..., 0]


            nll = recon_loss(flux, flux_err, mean)  # elementwise (B,L)
            observed_mask = mask.bool()  # 1 means observed; adapt if reversed
            loss = (nll * observed_mask).sum() / (observed_mask.sum().clamp_min(1.0))
            nll = recon_loss(flux, flux_err, mean)
            loss = nll.mean()
            # loss = (((mean - flux) ** 2).sum())**0.5

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n += 1

        avg = total_loss / max(1, n)
        print(f'Epoch {epoch+1}/{args.epochs} - Loss: {avg:.6f}')

    # Save final model
    outp = Path(args.output_dir)
    outp.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, outp / args.output_name)
    print('Saved final model to', outp / args.output_name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='data/timeseries.h5')
    p.add_argument('--max_length', type=int, default=16384)
    p.add_argument('--direction', type=str, default='forward', choices=['forward', 'backward', 'bi'])
    p.add_argument('--random_seed', type=int, default=19)
    p.add_argument('--num_samples', type=int, default=None)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--output_dir', type=str, default='output/simple_rnn')
    p.add_argument('--output_name', type=str, default='simple_min_gru_default_output.pt')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--use_flow', action='store_true')
    p.add_argument('--min_size', type=int, default=2)
    p.add_argument('--max_size', type=int, default=100)
    p.add_argument('--mask_portion', type=float, default=0.2)
    p.add_argument('--mock_sinusoid', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
