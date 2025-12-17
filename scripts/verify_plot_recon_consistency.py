import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lcgen.models.simple_min_gru import BiDirectionalMinGRU
from lcgen.train_simple_rnn import TimeSeriesDataset, collate_fn
from torch.utils.data import DataLoader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use an available checkpoint from output/simple_rnn/models
    model_path = 'output/simple_rnn/models/baseline_v5_test_refactor_seq.pt'
    model = BiDirectionalMinGRU(hidden_size=64, direction='bi').to(device)
    ckpt = torch.load(model_path)
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    try:
        model.load_state_dict(sd)
    except Exception:
        print('best-effort loading state dict')
        target = model.state_dict()
        if isinstance(sd, dict):
            for k, v in sd.items():
                if k in target and getattr(v, 'shape', None) == tuple(target[k].shape):
                    target[k] = v
        model.load_state_dict(target)
    model.eval()

    ds = TimeSeriesDataset(Path('data/timeseries.h5'), random_seed=19, min_size=2, max_size=40, mask_portion=0.6, max_length=256, num_samples=3)
    loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    flux, flux_err, times = batch

    flux = flux.to(device)
    flux_err = flux_err.to(device)
    x_in = torch.stack([flux, flux_err], dim=-1)
    t_in = times.to(device).unsqueeze(-1)

    with torch.no_grad():
        out = model(x_in, t_in, return_states=True)

    recon_model = out['reconstructed'].squeeze(-1).cpu().numpy()
    # prefer before-step hidden states (available prior to x_t)
    if 'h_fwd_before' not in out:
        raise RuntimeError("Model.forward must return 'h_fwd_before' when called with return_states=True")
    h_fwd = out['h_fwd_before']
    if model.direction == 'bi':
        if 'h_bwd_before' not in out:
            raise RuntimeError("Model.forward must return 'h_bwd_before' when model.direction=='bi' and return_states=True")
        h_bwd = out['h_bwd_before']
    else:
        h_bwd = out.get('h_bwd_before', None)
    t_enc_forward = out.get('t_enc')  # this was computed inside model.forward (shifted)

    # Manual reconstruction using forward's t_enc and hidden states
    B, L, H = h_fwd.shape
    Te = t_enc_forward.size(-1)
    manual_preds = np.full((B, L), np.nan)
    head = model.gauss_head
    for ti in range(L):
        if model.direction == 'bi' and h_bwd is not None:
            src_f = h_fwd[:, ti, :]
            src_b = h_bwd[:, ti, :]
            tgt = t_enc_forward[:, ti, :]
            inp = torch.cat([src_f, src_b, tgt], dim=1)
        else:
            src = h_fwd[:, ti, :]
            tgt = t_enc_forward[:, ti, :]
            inp = torch.cat([src, tgt], dim=1)
        flat = inp
        normed = model.head_norm(flat)
        if Te > 0:
            h_hidden = normed[:, :-Te]
            h_time = normed[:, -Te:] * model.time_scale
            normed = torch.cat([h_hidden, h_time], dim=1)
        p = head(normed).squeeze(-1).detach().cpu().numpy()
        manual_preds[:, ti] = p

    # Manual reconstruction using plot_recon-style computed t_enc from raw t_in
    with torch.no_grad():
        t_enc_in = model.time_enc(t_in)
    manual_preds_plotstyle = np.full((B, L), np.nan)
    for ti in range(L):
        if model.direction == 'bi' and h_bwd is not None:
            src_f = h_fwd[:, ti, :]
            src_b = h_bwd[:, ti, :]
            tgt = t_enc_in[:, ti, :]
            inp = torch.cat([src_f, src_b, tgt], dim=1)
        else:
            src = h_fwd[:, ti, :]
            tgt = t_enc_in[:, ti, :]
            inp = torch.cat([src, tgt], dim=1)
        flat = inp
        normed = model.head_norm(flat)
        if Te > 0:
            h_hidden = normed[:, :-Te]
            h_time = normed[:, -Te:] * model.time_scale
            normed = torch.cat([h_hidden, h_time], dim=1)
        p = head(normed).squeeze(-1).detach().cpu().numpy()
        manual_preds_plotstyle[:, ti] = p

    def summarize(a, b, name):
        diff = a - b
        mask = np.isfinite(diff)
        md = np.nanmax(np.abs(diff))
        mean = np.nanmean(np.abs(diff))
        print(f'{name}: max_abs_diff={md:.6e}, mean_abs_diff={mean:.6e}')

    print('Comparisons:')
    summarize(recon_model, manual_preds, 'model vs manual(using forward t_enc)')
    summarize(recon_model, manual_preds_plotstyle, 'model vs manual(using plot_recon t_enc)')
    summarize(manual_preds, manual_preds_plotstyle, 'manual(forward t_enc) vs manual(plot t_enc)')


if __name__ == "__main__":
    main()
