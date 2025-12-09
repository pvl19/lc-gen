import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
from pathlib import Path as _P
# Ensure local 'src' is on sys.path so we can import lcgen
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'src'))
from lcgen.models.simple_min_gru import SimpleMinGRU
from pathlib import Path
from lcgen.utils.trunc_data import extract_data
from lcgen.utils.loss import recon_loss
from torch.utils.data import DataLoader
from lcgen.models.simple_min_gru import SimpleMinGRU, BiDirectionalMinGRU
from lcgen.train_simple_rnn import TimeSeriesDataset, collate_fn

def mask_intervals(mask_1d):
    """
    Given a boolean array (L,) where True = masked, False = unmasked,
    return a list of (start, end) intervals that are masked.
    """
    masked = mask_1d.astype(bool)
    L = len(masked)

    intervals = []
    in_block = False
    start = None
    
    for i in range(L):
        if masked[i] and not in_block:
            in_block = True
            start = i
        elif not masked[i] and in_block:
            in_block = False
            intervals.append((start, i))
    if in_block:
        intervals.append((start, L))
    return intervals

def plot_recon(model_path, num_examples, seq_length, hidden_size, direction, use_flow, random_seed, mock_sinusoid):
    """Plot reconstructions from a trained SimpleMinGRU model.

    Args:
        model_path: Path to the trained model checkpoint (.pt file)
        num_examples: Number of examples to plot
        seq_length: Length of each time series
        random_seed: Random seed for reproducibility
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create folder for plots
    outp = Path(model_path).parent / 'recon_plots'
    outp.mkdir(parents=True, exist_ok=True)

    # Load model
    if direction == 'bi':
        model = BiDirectionalMinGRU(hidden_size=hidden_size, use_flow=use_flow)
    else:
        model = SimpleMinGRU(hidden_size=hidden_size, direction=direction, use_flow=use_flow)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    print('Loaded model from ', model_path)

    # Load data and generate reconstructions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_data_path = Path('data/timeseries.h5')
    ds = TimeSeriesDataset(raw_data_path, random_seed=random_seed, min_size=args.min_size, max_size=args.max_size, mask_portion=args.mask_portion, max_length=seq_length, num_samples=num_examples, mock_sinusoid=mock_sinusoid)
    loader = DataLoader(ds, batch_size=num_examples, shuffle=False, collate_fn=collate_fn)
    batch_X = next(iter(loader)) 
    flux, flux_err, times, mask = batch_X
    flux = flux.to(device)
    flux_err = flux_err.to(device)
    # masked_flux = masked_flux.to(device)
    # masked_flux_err = masked_flux_err.to(device)
    times = times.to(device)
    mask = mask.to(device)

    # Build input channels [flux, flux_err] -> (B, L, 2)
    x_in = torch.stack([flux, flux_err, mask], dim=-1)
    t_in = torch.stack([times], dim=-1)

    out = model(x_in, t_in)
    recon = out['reconstructed']  # (B, L, 1) -> [mean, raw_logsigma]
    mean = recon[..., 0]
    loss = recon_loss(flux, flux_err, mean)
    flux_np = flux.cpu().numpy()
    flux_err_np = flux_err.cpu().numpy()
    times_np = times.cpu().numpy()
    mean_np = mean.detach().cpu().numpy()
    loss_np = loss.detach().cpu().numpy()
    mask_np = mask.cpu().numpy()

    print(f'Using {num_examples} examples of length {seq_length} for plotting.')

    ### Plot
    fig, ax = plt.subplots(num_examples*2,1,figsize=(15, num_examples * 8))
    plt.tight_layout(pad=6.0)

    for i in range(num_examples):

        # Highlight masked intervals
        # Boolean mask for the i-th sample (True = masked)
        masked_positions = (mask_np[i] == 0)
        intervals = mask_intervals(masked_positions)

        # Reconstruction
        plt.subplot(num_examples*2,1,(i*2)+1)
        plt.errorbar(times_np[i], flux_np[i], yerr=flux_err_np[i], fmt='.', label='Original', alpha=0.5)
        plt.plot(times_np[i], mean_np[i], label='Reconstruction', color='red')
        for start, end in intervals:
            x0 = times_np[i, start]
            x1 = times_np[i, end-1]
            plt.axvspan(x0, x1, color='gray', alpha=0.2)
        plt.title(f'Example {i+1} - Reconstruction')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.legend()

        # Loss
        plt.subplot(num_examples*2,1,(i*2)+2)
        plt.plot(times_np[i], loss_np[i], label='Reconstruction Loss', color='green')
        for start, end in intervals:
            x0 = times_np[i, start]
            x1 = times_np[i, end-1]
            plt.axvspan(x0, x1, color='gray', alpha=0.2)
        plt.title(f'Example {i+1} - Reconstruction Loss')
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.legend()

    fig.savefig(outp / args.output_name)
    print(f'Saved reconstruction plots to {outp / args.output_name}')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, default='output/simple_rnn/simple_min_gru_final_dummymask.pt')
    p.add_argument('--seq_length', type=int, default=1024)
    p.add_argument('--output_name', type=str, default='reconstructions_rolling.png')
    p.add_argument('--direction', type=str, default='bi', choices=['forward','backward','bi'])
    p.add_argument('--num_examples', type=int, default=3)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--use_flow', action='store_true')   
    p.add_argument('--min_size', type=int, default=2)
    p.add_argument('--max_size', type=int, default=40)
    p.add_argument('--mask_portion', type=float, default=0.6)
    p.add_argument('--random_seed', type=int, default=19)
    p.add_argument('--mock_sinusoid', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    plot_recon(args.model_path, args.num_examples, args.seq_length, args.hidden_size, args.direction, args.use_flow, args.random_seed, args.mock_sinusoid)