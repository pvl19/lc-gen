#!/usr/bin/env python3
"""
Demo: load a trained RNN model and a few lightcurves, run the model with teacher forcing
and draw flow-based samples for visualization. Saves sampled traces to /tmp/flow_samples_demo.npz
"""
import sys
from pathlib import Path
import torch
import numpy as np

# allow imports from repo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_rnn_autoencoder import load_lightcurve_data, LightCurveDataset, collate_with_masking
from lcgen.models.rnn import HierarchicalRNN, RNNConfig

MODEL_PATH = Path('tmp_fulltrain_trunc2048_logprob/rnn_final.pt')
H5_PATH = Path('data/real_lightcurves/timeseries.h5')
# Save inside the project tmp/ directory (not root /tmp)
OUTPATH = Path('tmp/flow_samples_demo.npz')

BATCH_SIZE = 4
NUM_SAMPLES = 4

def load_model(device):
    cfg = RNNConfig()
    model = HierarchicalRNN(cfg).to(device)
    ckpt = torch.load(str(MODEL_PATH), map_location=device)
    # ckpt may be state_dict or dict containing keys
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
    else:
        # try treating loaded object as state_dict
        sd = ckpt
    try:
        model.load_state_dict(sd)
    except Exception as e:
        # try matching keys (strip 'module.' etc.)
        new_sd = {}
        for k, v in sd.items():
            new_k = k.replace('module.', '')
            new_sd[new_k] = v
        model.load_state_dict(new_sd)
    model.eval()
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if not MODEL_PATH.exists():
        print('Model file not found:', MODEL_PATH)
        return
    if not H5_PATH.exists():
        print('HDF5 data not found:', H5_PATH)
        return

    print('Loading model...')
    model = load_model(device)

    print('Loading data (first few examples)...')
    flux, timestamps, flux_err = load_lightcurve_data(str(H5_PATH))
    # take first BATCH_SIZE examples
    flux_small = flux[:BATCH_SIZE]
    time_small = timestamps[:BATCH_SIZE]

    # Build dataset items for collate function
    items = []
    for i in range(flux_small.shape[0]):
        items.append({'flux': torch.from_numpy(flux_small[i]).float(), 'time': torch.from_numpy(time_small[i]).float()})

    batch = collate_with_masking(items)
    x_input = batch['input']  # (B, L, 3)
    x_target = batch['target']
    mask = batch['mask']
    times = batch['time']

    print('Batch shapes:', x_input.shape, x_target.shape, mask.shape, times.shape)

    with torch.no_grad():
        base_out = model(x_input.to(device), times.to(device), target=x_target.to(device), mask=mask.to(device))

    if isinstance(base_out, dict) and base_out.get('flow_context', None) is not None and getattr(model, 'flow_posterior', None) is not None:
        flow_ctx = base_out['flow_context']  # (B, L, C)
        B, L, C = flow_ctx.shape
        ctx_flat = flow_ctx.view(B * L, -1)
        dist = model.flow_posterior(ctx_flat)
        try:
            s_all = dist.sample((NUM_SAMPLES,))
            s_all = s_all.view(NUM_SAMPLES, B, L)
        except TypeError:
            flat_list = []
            for idx in range(B * L):
                ctx_i = ctx_flat[idx:idx+1]
                dist_i = model.flow_posterior(ctx_i)
                si = dist_i.sample((NUM_SAMPLES,)).view(NUM_SAMPLES)
                flat_list.append(si)
            s_all = torch.stack(flat_list, dim=1).view(NUM_SAMPLES, B, L)

        sampled_np = s_all.cpu().numpy()
        print('Sampled shapes:', sampled_np.shape)
        np.savez_compressed(str(OUTPATH), sampled=sampled_np, target=x_target.numpy(), recon_mean=base_out.get('reconstructed', None))
        print('Saved sampled traces to', OUTPATH)
    else:
        print('No flow_posterior or flow_context available; cannot sample from flow.')

if __name__ == '__main__':
    main()
