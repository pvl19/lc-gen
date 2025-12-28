import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime

import sys
from pathlib import Path as _P
# Ensure local 'src' is on sys.path so we can import lcgen
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'src'))
from lcgen.models.simple_min_gru import SimpleMinGRU
from pathlib import Path
from lcgen.utils.trunc_data import extract_data
from lcgen.utils.loss import recon_loss
from lcgen.utils.run_log import log_run_args
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
    outp = Path(args.model_path).parent.parent / 'recon_plots'
    outp_data = Path(args.model_path).parent.parent / 'recon_data'
    outp.mkdir(parents=True, exist_ok=True)
    outp_data.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiDirectionalMinGRU(hidden_size=args.hidden_size, direction=args.direction, mode=args.mode, use_flow=args.use_flow).to(device)
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
    if getattr(model, 'flow', None) is not None:
        print('Flow head is enabled in model (zuko flow present).')
    else:
        print('Flow head not present in model; using Gaussian head for reconstructions.')

    # Load data and generate reconstructions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_data_path = Path('data/timeseries.h5')
    ds = TimeSeriesDataset(raw_data_path, random_seed=args.random_seed, min_size=args.min_size, max_size=args.max_size, mask_portion=args.mask_portion, max_length=args.seq_length, num_samples=args.num_examples, mock_sinusoid=args.mock_sinusoid, mock_noise=args.mock_noise)
    loader = DataLoader(ds, batch_size=args.num_examples, shuffle=False, collate_fn=collate_fn)
    batch_X = next(iter(loader)) 
    flux, flux_err, times, mask = batch_X
    flux = flux.to(device)
    flux_err = flux_err.to(device)
    times = times.to(device)
    mask = mask.to(device)

    # Build input channels [flux, flux_err] -> (B, L, 2)
    x_in = torch.stack([flux, flux_err], dim=-1)
    t_in = torch.stack([times], dim=-1)

    # Pass flow_mode to control how flow head produces point predictions
    flow_mode = getattr(args, 'flow_mode', 'mean')
    print(f'Flow mode for reconstruction: {flow_mode}')
    out = model(x_in, t_in, mask=mask, return_states=True, flow_mode=flow_mode)
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
    mask_np = mask.cpu().numpy()

    # Prepare multi-horizon reconstructions using hidden states at t-k
    if args.lags is None:
        ks = [1, 2, 8, 32]
    else:
        ks = list(args.lags)
    preds_ks = {}
    preds_ks_p16 = {}  # 16th percentile of MC samples
    preds_ks_p84 = {}  # 84th percentile of MC samples
    if h_fwd is not None and t_enc_in is not None:
        # We'll also compute predictions using relative time encodings (t_target - t_source)
        preds_ks_rel = {}
        preds_ks_loss = {}

        # h_fwd: (B, L, H), t_enc_in: (B, L, Te)
        B, L, H = h_fwd.shape
        Te = t_enc_in.size(-1)
        head = model.gauss_head
        flow_head = getattr(model, 'flow', None)
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

                    # measurement error at target positions
                    ferr_tgt = flux_err[:, k:]
                    ferr_flat = ferr_tgt.contiguous().view(-1, 1)

                    # If flow head is present, use flow_mode to produce predictions
                    if flow_head is not None:
                        ctx = torch.cat([normed, ferr_flat], dim=1)
                        dist = flow_head(ctx)
                        
                        if flow_mode == 'sample':
                            # Single random sample
                            flat_preds = dist.sample().view(B, L - k)
                            preds_p16 = None
                            preds_p84 = None
                        elif flow_mode == 'mode':
                            # Gradient-based optimization to find the mode (MAP estimate)
                            N = ctx.size(0)
                            with torch.no_grad():
                                y_init = dist.sample().clone()
                            y_opt = y_init.requires_grad_(True)
                            optimizer = torch.optim.Adam([y_opt], lr=0.1)
                            for _ in range(50):
                                optimizer.zero_grad()
                                loss = -dist.log_prob(y_opt).sum()
                                loss.backward()
                                optimizer.step()
                                with torch.no_grad():
                                    y_opt.clamp_(-10, 10)
                            flat_preds = y_opt.detach().view(B, L - k)
                            preds_p16 = None
                            preds_p84 = None
                        else:  # 'mean'
                            # Collect multiple samples to approximate the conditional mean and percentiles
                            n_samples = 32
                            samples = torch.stack([dist.sample() for _ in range(n_samples)], dim=0)  # (n_samples, B*(L-k), 1)
                            samples = samples.squeeze(-1).view(n_samples, B, L - k)  # (n_samples, B, L-k)
                            flat_preds = samples.mean(dim=0)  # (B, L-k)
                            preds_p16 = torch.quantile(samples, 0.16, dim=0)  # (B, L-k)
                            preds_p84 = torch.quantile(samples, 0.84, dim=0)  # (B, L-k)
                    else:
                        flat_preds = head(normed).view(B, L - k)
                        preds_p16 = None
                        preds_p84 = None
                    preds_k[:, k:] = flat_preds
                    # Store percentiles if computed
                    if preds_p16 is not None:
                        preds_k_p16 = torch.full((B, L), float('nan'), device=device)
                        preds_k_p84 = torch.full((B, L), float('nan'), device=device)
                        preds_k_p16[:, k:] = preds_p16
                        preds_k_p84[:, k:] = preds_p84
                        preds_ks_p16[k] = preds_k_p16.cpu().numpy()
                        preds_ks_p84[k] = preds_k_p84.cpu().numpy()
                    # compute per-prediction NLL aligned with target indices
                    with torch.no_grad():
                        preds_k_t = preds_k[:, k:]
                        flux_tgt = flux[:, k:]
                        # Use flow log_prob for NLL if flow is present
                        if flow_head is not None:
                            flux_flat = flux_tgt.contiguous().view(-1, 1)
                            ctx = torch.cat([normed, ferr_flat], dim=1)
                            dist = flow_head(ctx)
                            logp = dist.log_prob(flux_flat)
                            if logp.dim() > 1:
                                logp = logp.view(-1)
                            nll_k = -logp.view(B, L - k)
                        else:
                            nll_k = recon_loss(flux_tgt, ferr_tgt, preds_k_t)  # (B, L-k)
                        loss_k = torch.full((B, L), float('nan'), device=device)
                        loss_k[:, k:] = nll_k
                        preds_ks_loss[k] = loss_k.cpu().numpy()
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

                    # Also use flow for relative-time predictions if present
                    if flow_head is not None:
                        ferr_tgt_rel = flux_err[:, k:]
                        ferr_flat_rel = ferr_tgt_rel.contiguous().view(-1, 1)
                        ctx_rel = torch.cat([normed_rel, ferr_flat_rel], dim=1)
                        dist_rel = flow_head(ctx_rel)
                        
                        if flow_mode == 'sample':
                            flat_preds_rel = dist_rel.sample().view(B, L - k)
                        elif flow_mode == 'mode':
                            N_rel = ctx_rel.size(0)
                            with torch.no_grad():
                                y_init_rel = dist_rel.sample().clone()
                            y_opt_rel = y_init_rel.requires_grad_(True)
                            optimizer_rel = torch.optim.Adam([y_opt_rel], lr=0.1)
                            for _ in range(50):
                                optimizer_rel.zero_grad()
                                loss = -dist_rel.log_prob(y_opt_rel).sum()
                                loss.backward()
                                optimizer_rel.step()
                                with torch.no_grad():
                                    y_opt_rel.clamp_(-10, 10)
                            flat_preds_rel = y_opt_rel.detach().view(B, L - k)
                        else:  # 'mean'
                            n_samples = 16
                            s_acc = dist_rel.sample()
                            for _ in range(n_samples - 1):
                                s_acc = s_acc + dist_rel.sample()
                            flat_preds_rel = (s_acc / n_samples).view(B, L - k)
                    else:
                        flat_preds_rel = head(normed_rel).view(B, L - k)
                    preds_k_rel[:, k:] = flat_preds_rel
                preds_ks_rel[k] = preds_k_rel.cpu().numpy()
    else:
        # fallback: repeat the single-step mean for all ks
        print('Warning: no hidden states or time encodings available; using single-step mean for all ks.')
        for k in ks:
            preds_ks[k] = mean_np
            preds_ks_rel[k] = mean_np
            preds_ks_loss[k] = np.zeros_like(mean_np)

    print(f'Using {args.num_examples} examples of length {args.seq_length} for plotting.')

    ### Plot
    fig, ax = plt.subplots(args.num_examples*2,1,figsize=(15, args.num_examples * 8))
    plt.tight_layout(pad=6.0)

    for i in range(args.num_examples):

        # Highlight masked intervals
        # Boolean mask for the i-th sample (True = masked, i.e., mask_np == 0)
        masked_positions = (mask_np[i] == 0)
        intervals = mask_intervals(masked_positions)

        # Reconstruction: plot true flux and multiple reconstructions from different lagged states
        plt.subplot(args.num_examples*2, 1, (i*2) + 1)
        plt.errorbar(times_np[i], flux_np[i], yerr=flux_err_np[i], fmt='.', label='Original', alpha=0.5, color='black',
                     ecolor='lightgray')
        # plot each k prediction if available; pick colors from a colormap
        cmap = plt.get_cmap('tab10')
        for idx, k in enumerate(ks):
            pred_k = preds_ks.get(k)
            if pred_k is None:
                continue
            pred_k_i = pred_k[i]
            color = cmap(idx % 10)
            # pred_k may contain NaNs for early timesteps; matplotlib will skip them
            # Mean as solid line
            plt.plot(times_np[i], pred_k_i, label=f'Pred from t-{k}', color=color, linestyle='-', linewidth=2)
            # Plot 16th and 84th percentiles as dashed lines if available
            pred_p16 = preds_ks_p16.get(k)
            pred_p84 = preds_ks_p84.get(k)
            if pred_p16 is not None and pred_p84 is not None:
                plt.plot(times_np[i], pred_p16[i], color=color, linestyle='--', linewidth=1, alpha=0.7)
                plt.plot(times_np[i], pred_p84[i], color=color, linestyle='--', linewidth=1, alpha=0.7)
        # Highlight masked regions
        for start, end in intervals:
            x0 = times_np[i, start]
            x1 = times_np[i, end-1] if end > start else x0
            plt.axvspan(x0, x1, color='red', alpha=0.15, label='Masked' if start == intervals[0][0] else None)
        plt.title(f'Example {i+1} - Reconstruction (multi-horizon)')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.legend()

        # Loss: plot per-k NLL lines (use same colors as predictions)
        plt.subplot(args.num_examples*2, 1, (i*2) + 2)
        cmap = plt.get_cmap('tab10')
        for idx, k in enumerate(ks):
            loss_arr = preds_ks_loss.get(k)
            if loss_arr is None:
                continue
            loss_i = loss_arr[i]
            color = cmap(idx % 10)
            plt.plot(times_np[i], loss_i, label=f'Loss from pred t-{k}', color=color)
        # Highlight masked regions
        for start, end in intervals:
            x0 = times_np[i, start]
            x1 = times_np[i, end-1] if end > start else x0
            plt.axvspan(x0, x1, color='red', alpha=0.15)
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
        'mask': mask_np,
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
    for k, arr in preds_ks_loss.items():
        save_dict[f'loss_k_{k}'] = arr
    for k, arr in preds_ks_rel.items():
        save_dict[f'pred_k_rel_{k}'] = arr
    for k, arr in preds_ks_p16.items():
        save_dict[f'pred_k_p16_{k}'] = arr
    for k, arr in preds_ks_p84.items():
        save_dict[f'pred_k_p84_{k}'] = arr

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
    p.add_argument('--mock_noise', type=float, default=0.1)
    p.add_argument('--mode', type=str, default='sequential', choices=['sequential', 'parallel'])
    p.add_argument('--lags', type=int, nargs='*', default=None, help='List of lag values (t-k) for multi-horizon reconstructions')
    p.add_argument('--flow_mode', type=str, default='mean', choices=['mode', 'mean', 'sample'],
                   help='How to produce point predictions from flow head: mode (MAP), mean (MC average), sample (single random)')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    log_run_args(args, log_file='output/simple_rnn/logs/plot_recon_log.json')
    print('Starting at time', datetime.datetime.now())
    plot_recon(args)