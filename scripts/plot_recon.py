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
    outp_data = Path(args.model_path).parent / 'recon_data'
    outp.mkdir(parents=True, exist_ok=True)
    outp_data.mkdir(parents=True, exist_ok=True)

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
    # Compute time-encodings from the provided raw times but first shift each
    # sequence so it starts at zero: t_shifted = t_in - t0 (t0 = first time per seq).
    # This uses the user's provided time base while matching the model's forward
    # convention of computing time encodings on a per-sequence zero-origin.
    with torch.no_grad():
        # t_in shape: (B, L, 1) -> squeeze to (B, L)
        t_seq = t_in.squeeze(-1)
        t0 = t_seq[:, 0].unsqueeze(1)  # (B, 1)
        t_shifted = (t_seq - t0).unsqueeze(-1)  # (B, L, 1)
        t_enc_in = model.time_enc(t_shifted)
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
        # We'll also compute predictions using relative time encodings (t_target - t_source)
        preds_ks_rel = {}

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
                # ---- relative time predictions for this k ----
                preds_k_rel = torch.full((B, L), float('nan'), device=device)
                if k < L:
                    # build relative time differences: shape (B, L-k, 1)
                    # t_src: times[:, :L-k], t_tgt: times[:, k:]
                    t_src = times[:, :L-k]
                    t_tgt = times[:, k:]
                    t_rel = (t_tgt - t_src).unsqueeze(-1)  # (B, L-k, 1)
                    # compute time-encodings from relative times
                    t_rel_enc = model.time_enc(t_rel)
                    if args.direction == 'bi' and h_bwd is not None:
                        src_f = h_fwd[:, :L-k, :]
                        src_b = h_bwd[:, k:, :]
                        tgt_rel = t_rel_enc
                        inputs_k_rel = torch.cat([src_f, src_b, tgt_rel], dim=-1)
                    else:
                        src = h_fwd[:, :L-k, :]
                        tgt_rel = t_rel_enc
                        inputs_k_rel = torch.cat([src, tgt_rel], dim=-1)

                    flat_in_rel = inputs_k_rel.contiguous().view(-1, inputs_k_rel.size(-1))
                    normed_rel = model.head_norm(flat_in_rel)
                    if nt > 0:
                        h_hidden = normed_rel[:, :-nt]
                        h_time = normed_rel[:, -nt:] * model.time_scale
                        normed_rel = torch.cat([h_hidden, h_time], dim=1)
                    flat_preds_rel = head(normed_rel).view(B, L - k)
                    preds_k_rel[:, k:] = flat_preds_rel
                preds_ks_rel[k] = preds_k_rel.cpu().numpy()
    else:
        # fallback: repeat the single-step mean for all ks
        print('Warning: no hidden states or time encodings available; using single-step mean for all ks.')
        for k in ks:
            preds_ks[k] = mean_np
            preds_ks_rel[k] = mean_np

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
    # Save plotting data (numpy) for later analysis / comparisons
    data_fname = args.output_name.replace('.png', '') + '_data.npz'
    save_path = outp_data / data_fname
    # Build a dict of arrays to save
    save_dict = {
        'flux': flux_np,
        'flux_err': flux_err_np,
        'times': times_np,
        'mean': mean_np,
        'loss': loss_np,
        # include the time-encodings computed from the input times so we can
        # inspect how time features vary and contribute to per-k differences
        't_enc_in': t_enc_in.detach().cpu().numpy(),
        'ks': np.array(ks, dtype=np.int32)
    }
    for k, arr in preds_ks.items():
        # store each preds_k under a key like pred_k_1
        save_dict[f'pred_k_{k}'] = arr
    for k, arr in preds_ks_rel.items():
        save_dict[f'pred_k_rel_{k}'] = arr

    try:
        np.savez_compressed(save_path, **save_dict)
        print(f'Saved reconstruction plot to {outp / args.output_name} and data to {save_path}')
    except Exception as e:
        print(f'Warning: failed to save data npz to {save_path}: {e}')

    # Also save relative-time predictions to a separate NPZ for direct comparison
    data_fname_rel = args.output_name.replace('.png', '') + '_rel_data.npz'
    save_path_rel = outp_data / data_fname_rel
    save_dict_rel = save_dict.copy()
    for k, arr in preds_ks_rel.items():
        save_dict_rel[f'pred_k_rel_{k}'] = arr
    try:
        np.savez_compressed(save_path_rel, **save_dict_rel)
        print(f'Saved relative-time reconstruction data to {save_path_rel}')
    except Exception as e:
        print(f'Warning: failed to save relative data npz to {save_path_rel}: {e}')
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