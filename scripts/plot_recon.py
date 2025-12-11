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

def plot_recon(args):
    """Plot reconstructions from a trained SimpleMinGRU model.

    Args:
        args: parsed argparse Namespace with expected attributes
    """
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Create folder for plots
    outp = Path(args.model_path).parent / 'recon_plots'
    outp.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiDirectionalMinGRU(hidden_size=args.hidden_size, direction=args.direction, use_flow=args.use_flow).to(device)
    ckpt = torch.load(args.model_path)
    # ckpt may store a dict under 'model_state_dict'
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
    try:
        model.load_state_dict(sd)
    except Exception:
        # best-effort: load matching keys only
        print('Warning: loading model state dict with best-effort key matching.')
        target = model.state_dict()
        if isinstance(sd, dict):
            for k, v in sd.items():
                if k in target and getattr(v, 'shape', None) == tuple(target[k].shape):
                    target[k] = v
        model.load_state_dict(target)
    model.eval()
    print('Loaded model from ', args.model_path)

    # Load data and generate reconstructions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_data_path = Path('data/timeseries.h5')
    ds = TimeSeriesDataset(raw_data_path, random_seed=args.random_seed, min_size=args.min_size, max_size=args.max_size, mask_portion=args.mask_portion, max_length=args.seq_length, num_samples=args.num_examples, mock_sinusoid=args.mock_sinusoid)
    loader = DataLoader(ds, batch_size=args.num_examples, shuffle=False, collate_fn=collate_fn)
    batch_X = next(iter(loader)) 
    flux, flux_err, times = batch_X
    flux = flux.to(device)
    flux_err = flux_err.to(device)
    # masked_flux = masked_flux.to(device)
    # masked_flux_err = masked_flux_err.to(device)
    times = times.to(device)
    # mask = mask.to(device)

    # Build input channels [flux, flux_err] -> (B, L, 2)
    x_in = torch.stack([flux, flux_err], dim=-1)
    t_in = torch.stack([times], dim=-1)

    out = model(x_in, t_in, return_states=True)
    recon = out['reconstructed']  # (B, L, 1) -> [mean]
    mean = recon[..., 0]
    # get forward/backward hidden states from the model output
    h_fwd = out.get('h_fwd_tensor')
    h_bwd_out = out.get('h_bwd_tensor')
    # Compute time-encodings directly from the input times (t_in) so that
    # reconstructions for t+k are formed by concatenating hidden(t) with
    # time_enc(t+k) computed from the original input times. Do NOT rely on
    # t_enc returned by the forward pass (it may reflect internal normalization
    # or checkpoint mismatches); compute from `t_in` explicitly.
    with torch.no_grad():
        t_enc_in = model.time_enc(t_in)
    loss = recon_loss(flux, flux_err, mean)
    flux_np = flux.cpu().numpy()
    flux_err_np = flux_err.cpu().numpy()
    times_np = times.cpu().numpy()
    mean_np = mean.detach().cpu().numpy()
    loss_np = loss.detach().cpu().numpy()
    # mask_np = mask.cpu().numpy()

    # Prepare multi-horizon reconstructions using hidden states at t-k
    if args.lags is None:
        ks = [1, 2, 8, 32]
    else:
        ks = list(args.lags)
    preds_ks = {}
    if h_fwd is not None and t_enc_in is not None:
        # h_fwd: (B, L, H), t_enc_in: (B, L, Te)
        B, L, H = h_fwd.shape
        Te = t_enc_in.size(-1)
        head = model.gauss_head
        h_bwd = h_bwd_out
        with torch.no_grad():
            for k in ks:
                preds_k = torch.full((B, L), float('nan'), device=device)
                if k < L:
                    if args.direction == 'bi' and h_bwd is not None:
                        # For bi-directional model, use forward hidden at source (i=j-k)
                        # and backward hidden at target (j) so shapes align: src_f: (B,L-k,H), src_b: (B,L-k,H), tgt: (B,L-k,Te)
                        src_f = h_fwd[:, :L-k, :]
                        src_b = h_bwd[:, k:, :]
                        # use time-encodings computed from input times
                        tgt = t_enc_in[:, k:, :]
                        inputs_k = torch.cat([src_f, src_b, tgt], dim=-1)   # (B, L-k, 2H+Te)
                    else:
                        # forward or backward single-direction model: use single hidden + time enc
                        src = h_fwd[:, :L-k, :]
                        tgt = t_enc_in[:, k:, :]
                        inputs_k = torch.cat([src, tgt], dim=-1)   # (B, L-k, H+Te)

                    flat_in = inputs_k.contiguous().view(-1, inputs_k.size(-1))
                    # apply head normalization and post-LN time scaling to match model.forward
                    normed = model.head_norm(flat_in)
                    nt = Te
                    if nt > 0:
                        h_hidden = normed[:, :-nt]
                        h_time = normed[:, -nt:] * model.time_scale
                        normed = torch.cat([h_hidden, h_time], dim=1)
                    flat_preds = head(normed).view(B, L - k)
                    preds_k[:, k:] = flat_preds
                preds_ks[k] = preds_k.cpu().numpy()
    else:
        # fallback: repeat the single-step mean for all ks
        print('Warning: no hidden states or time encodings available; using single-step mean for all ks.')
        for k in ks:
            preds_ks[k] = mean_np

    print(f'Using {args.num_examples} examples of length {args.seq_length} for plotting.')

    ### Plot
    fig, ax = plt.subplots(args.num_examples*2,1,figsize=(15, args.num_examples * 8))
    plt.tight_layout(pad=6.0)

    for i in range(args.num_examples):

        # Highlight masked intervals
        # Boolean mask for the i-th sample (True = masked)
        # masked_positions = (mask_np[i] == 0)
        # intervals = mask_intervals(masked_positions)

        # Reconstruction: plot true flux and multiple reconstructions from different lagged states
        plt.subplot(args.num_examples*2, 1, (i*2) + 1)
        plt.errorbar(times_np[i], flux_np[i], yerr=flux_err_np[i], fmt='.', label='Original', alpha=0.5, color='black')
        # plt.plot(times_np[i], mean_np[i], label='Reconstruction (t-1 default)', color='black', linewidth=1.5)
        # plot each k prediction if available; pick colors from a colormap
        cmap = plt.get_cmap('tab10')
        for idx, k in enumerate(ks):
            pred_k = preds_ks.get(k)
            if pred_k is None:
                continue
            pred_k_i = pred_k[i]
            color = cmap(idx % 10)
            # pred_k may contain NaNs for early timesteps; matplotlib will skip them
            plt.plot(times_np[i], pred_k_i, label=f'Pred from t-{k}', color=color, linestyle='--')
            # for start, end in intervals:
            #     x0 = times_np[i, start]
            #     x1 = times_np[i, end-1]
            #     plt.axvspan(x0, x1, color='gray', alpha=0.2)
        plt.title(f'Example {i+1} - Reconstruction (multi-horizon)')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.legend()

        # Loss
        plt.subplot(args.num_examples*2, 1, (i*2) + 2)
        plt.plot(times_np[i], loss_np[i], label='Reconstruction Loss', color='green')
        # for start, end in intervals:
        #     x0 = times_np[i, start]
        #     x1 = times_np[i, end-1]
        #     plt.axvspan(x0, x1, color='gray', alpha=0.2)
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
    p.add_argument('--lags', type=int, nargs='*', default=None, help='List of lag values (t-k) for multi-horizon reconstructions')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    plot_recon(args)