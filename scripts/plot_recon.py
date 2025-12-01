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
from lcgen.models.simple_min_gru import SimpleMinGRU
from lcgen.train_simple_rnn import TimeSeriesDataset, collate_fn

def plot_recon(model_path, num_examples, seq_length, hidden_size, use_flow, random_seed=19):
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
    model = SimpleMinGRU(hidden_size=hidden_size, use_flow=use_flow)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    print('Loaded model from ', model_path)

    # Load data and generate reconstructions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_data_path = Path('data/timeseries.h5')
    ds = TimeSeriesDataset(raw_data_path, random_seed=random_seed, max_length=seq_length, num_samples=num_examples)
    loader = DataLoader(ds, batch_size=num_examples, shuffle=False, collate_fn=collate_fn)
    batch_X = next(iter(loader)) 
    flux, flux_err, times = batch_X
    flux = flux.to(device)
    flux_err = flux_err.to(device)
    times = times.to(device)

    # Build input channels [flux, flux_err] -> (B, L, 2)
    x_in = torch.stack([flux, flux_err], dim=-1)

    out = model(x_in)
    recon = out['reconstructed']  # (B, L, 1) -> [mean, raw_logsigma]
    mean = recon[..., 0]
    loss = recon_loss(flux, flux_err, mean)
    flux_np = flux.cpu().numpy()
    flux_err_np = flux_err.cpu().numpy()
    times_np = times.cpu().numpy()
    mean_np = mean.detach().cpu().numpy()
    loss_np = loss.detach().cpu().numpy()

    print(f'Using {num_examples} examples of length {seq_length} for plotting.')

    ### Plot
    fig, ax = plt.subplots(num_examples*2,1,figsize=(15, num_examples * 8))
    plt.tight_layout(pad=6.0)

    for i in range(num_examples):

        # Reconstruction
        plt.subplot(num_examples*2,1,(i*2)+1)
        plt.errorbar(times_np[i], flux_np[i], yerr=flux_err_np[i], fmt='.', label='Original', alpha=0.5)
        plt.plot(times_np[i], mean_np[i], label='Reconstruction', color='red')
        plt.title(f'Example {i+1} - Reconstruction')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.legend()

        # Loss
        plt.subplot(num_examples*2,1,(i*2)+2)
        plt.plot(times_np[i], loss_np[i], label='Reconstruction Loss', color='green')
        plt.title(f'Example {i+1} - Reconstruction Loss')
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.legend()

    
    fig.savefig(outp / 'reconstructions.png')
    print(f'Saved reconstruction plots to {outp / "reconstructions.png"}')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, default='output/simple_rnn/simple_min_gru_final.pt')
    p.add_argument('--seq_length', type=int, default=1024)
    p.add_argument('--num_examples', type=int, default=3)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--use_flow', action='store_true')   
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    plot_recon(args.model_path, args.num_examples, args.seq_length, args.hidden_size, args.use_flow)