#!/usr/bin/env python3
"""Visualize learned time encodings from the model.

Produces:
 - heatmap of encoding dimensions vs sorted raw time
 - per-dimension scatter/line plots vs raw time (first N dims)
 - PCA 2D scatter of encodings colored by raw time
 - bar plot of Pearson/Spearman correlation per encoding dim with raw time

Usage:
 python scripts/visualize_time_enc.py --model_path path/to/model.pt --num_samples 200 --seq_length 1024 --output_dir viz_timeenc --dims 8

This script avoids heavy external deps: uses numpy SVD for PCA and fallback rank implementation for Spearman.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure local src on path
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'src'))
from lcgen.models.simple_min_gru import BiDirectionalMinGRU
from lcgen.models.TimeSeriesDataset import TimeSeriesDataset, collate_fn

# small utility: compute ranks (1..n) using numpy argsort twice
def rankdata(a):
    # return ranks starting at 1
    tmp = np.argsort(a)
    ranks = np.empty_like(tmp)
    ranks[tmp] = np.arange(len(a))
    return ranks.astype(float) + 1.0

# pearson and spearman
def pearsonr(x, y):
    xm = x - x.mean()
    ym = y - y.mean()
    num = (xm * ym).sum()
    den = np.sqrt((xm**2).sum() * (ym**2).sum())
    if den == 0:
        return 0.0
    return float(num / den)

def spearmanr(x, y):
    rx = rankdata(x)
    ry = rankdata(y)
    return pearsonr(rx, ry)

# PCA via SVD (returns projected 2D)
def pca_2d(X, n_components=2):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:n_components]
    proj = Xc.dot(comp.T)
    return proj


def collect_time_encodings(model, ds, device, num_samples=200, seq_length=1024):
    # gather up to num_samples sequences and return flattened time encs and raw times
    loader = torch.utils.data.DataLoader(ds, batch_size=min(32, num_samples), shuffle=False, collate_fn=collate_fn)
    all_t = []
    all_tenc = []
    collected = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            flux, flux_err, times = batch
            B, L = flux.shape
            if L < 2:
                continue
            x_in = torch.stack([flux, flux_err], dim=-1).to(device)
            t_in = times.unsqueeze(-1).to(device)
            out = model(x_in, t_in, return_states=True)
            t_enc = out.get('t_enc')   # (B, L, Te)
            if t_enc is None:
                # try computing directly
                t_enc = model.time_enc(t_in.to(device))
            # flatten
            t_np = times.numpy().reshape(-1)
            tenc_np = t_enc.cpu().numpy().reshape(-1, t_enc.size(-1))
            all_t.append(t_np)
            all_tenc.append(tenc_np)
            collected += B
            if collected >= num_samples:
                break
    if len(all_t) == 0:
        return None, None
    t_flat = np.concatenate(all_t, axis=0)
    tenc_flat = np.concatenate(all_tenc, axis=0)
    return t_flat, tenc_flat


def plot_time_enc(t, tenc, output_dir: Path, dims: int = 8, prefix: str = 'timeenc'):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sort by raw time to help visualization
    order = np.argsort(t)
    t_sorted = t[order]
    tenc_sorted = tenc[order]

    Te = tenc.shape[1]
    dims = min(dims, Te)

    # Heatmap of all dimensions vs time (sorted)
    plt.figure(figsize=(10, 6))
    plt.imshow(tenc_sorted.T, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='encoding value')
    plt.xlabel('ordered sample index by time')
    plt.ylabel('encoding dimension')
    plt.title('Time encoding heatmap (dims x ordered time)')
    plt.savefig(output_dir / f'{prefix}_heatmap.png', dpi=150)
    plt.close()

    # Per-dimension scatter + rolling mean
    plt.figure(figsize=(12, 2 * dims))
    for i in range(dims):
        plt.subplot(dims, 1, i+1)
        plt.scatter(t, tenc[:, i], s=6, alpha=0.6)
        # simple moving average for trend
        order_idx = np.argsort(t)
        vals_sorted = tenc[order_idx, i]
        window = max(5, len(vals_sorted)//100)
        if window > 1:
            cumsum = np.cumsum(np.insert(vals_sorted, 0, 0))
            ma = (cumsum[window:] - cumsum[:-window]) / window
            x_ma = t_sorted[window-1:]
            plt.plot(x_ma, ma, color='red', linewidth=1)
        plt.title(f'dim {i} vs time')
        if i == dims-1:
            plt.xlabel('time')
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_per_dim.png', dpi=150)
    plt.close()

    # PCA scatter
    proj = pca_2d(tenc)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(proj[:,0], proj[:,1], c=t, cmap='plasma', s=8)
    plt.colorbar(sc, label='raw time')
    plt.title('PCA(2) of time encodings colored by raw time')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.savefig(output_dir / f'{prefix}_pca2.png', dpi=150)
    plt.close()

    # Correlations
    pear = [pearsonr(t, tenc[:, i]) for i in range(Te)]
    spear = [spearmanr(t, tenc[:, i]) for i in range(Te)]
    plt.figure(figsize=(10,4))
    plt.bar(np.arange(Te) - 0.2, pear, width=0.4, label='pearson')
    plt.bar(np.arange(Te) + 0.2, spear, width=0.4, label='spearman')
    plt.xlabel('encoding dim'); plt.ylabel('correlation with raw time')
    plt.legend(); plt.title('Correlation between encoding dims and raw time')
    plt.savefig(output_dir / f'{prefix}_correlations.png', dpi=150)
    plt.close()

    # Print top correlated dims
    pear_abs = np.abs(pear)
    top_idx = np.argsort(-pear_abs)[:10]
    print('Top dims by absolute Pearson correlation:', top_idx.tolist())
    for idx in top_idx[:10]:
        print(f'dim {idx}: pearson={pear[idx]:.4f}, spearman={spear[idx]:.4f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--seq_length', type=int, default=1024)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--direction', type=str, default='bi')
    p.add_argument('--use_flow', action='store_true')
    p.add_argument('--random_seed', type=int, default=19)
    p.add_argument('--output_dir', type=str, default='viz_timeenc')
    p.add_argument('--dims', type=int, default=8)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint first to infer the architecture used when saving
    ckpt = torch.load(args.model_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)

    # Infer time encoding dimensionality from checkpoint if present
    if 'time_enc.0.weight' in state:
        num_time_enc_dims = state['time_enc.0.weight'].shape[0]
    else:
        num_time_enc_dims = 64

    # Infer hidden size from forward_cell weights if present
    if 'forward_cell.W_z.weight' in state:
        hidden_size_ckpt = state['forward_cell.W_z.weight'].shape[0]
    elif 'forward_input_proj.weight' in state:
        hidden_size_ckpt = state['forward_input_proj.weight'].shape[0]
    else:
        hidden_size_ckpt = args.hidden_size

    # Infer direction by looking at gauss_head input dim
    gh0 = state.get('gauss_head.0.weight')
    if gh0 is not None:
        gh_in = gh0.shape[1]
        if gh_in == hidden_size_ckpt + num_time_enc_dims:
            direction_ckpt = 'forward'
        elif gh_in == 2 * hidden_size_ckpt + num_time_enc_dims:
            direction_ckpt = 'bi'
        else:
            direction_ckpt = args.direction
    else:
        direction_ckpt = args.direction

    # Instantiate model with inferred hidden size and direction, then override
    # submodules that depend on num_time_enc_dims so shapes match the checkpoint.
    model = BiDirectionalMinGRU(hidden_size=hidden_size_ckpt, direction=direction_ckpt, use_flow=args.use_flow).to(device)

    # Override time encoder and input projection layers to match checkpoint dims
    model.time_enc = torch.nn.Sequential(
        torch.nn.Linear(1, num_time_enc_dims),
        torch.nn.ReLU(),
        torch.nn.Linear(num_time_enc_dims, num_time_enc_dims),
    )
    model.forward_input_proj = torch.nn.Linear(2 + num_time_enc_dims, hidden_size_ckpt)
    model.backward_input_proj = torch.nn.Linear(2 + num_time_enc_dims, hidden_size_ckpt)

    # Recreate gauss_head with the correct input size depending on direction
    if direction_ckpt == 'bi':
        output_size = hidden_size_ckpt * 2 + num_time_enc_dims
    else:
        output_size = hidden_size_ckpt + num_time_enc_dims
    head_hidden = max(32, hidden_size_ckpt // 2)
    model.gauss_head = torch.nn.Sequential(
        torch.nn.Linear(output_size, head_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(head_hidden, 1),
    )

    # Now load the checkpoint (map to device after shapes are compatible)
    model.load_state_dict(state)
    model.to(device)
    print('Loaded model', args.model_path, '(inferred time_enc dims=', num_time_enc_dims, 'hidden=', hidden_size_ckpt, 'direction=', direction_ckpt, ')')

    # Prepare dataset
    ds = TimeSeriesDataset('data/timeseries.h5', random_seed=args.random_seed, min_size=2, max_size=40, mask_portion=0.0, max_length=args.seq_length, num_samples=args.num_samples)

    t, tenc = collect_time_encodings(model, ds, device, num_samples=args.num_samples, seq_length=args.seq_length)
    if t is None:
        print('No data collected; check dataset or parameters')
        return

    outdir = Path(args.output_dir)
    plot_time_enc(t, tenc, outdir, dims=args.dims)

if __name__ == '__main__':
    main()
