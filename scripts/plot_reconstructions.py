"""Plot reconstructions of light curves from a trained BiDirectionalMinGRU + flow model.

Computes hidden states for selected light curves and predicts flux at a user-specified
offset k (in timesteps) using the flow head. Shows measured flux with error bars plus
the flow median and a translucent p16-p84 band.

Examples:
  python plot_reconstructions.py --model_path final_model/final_parallel_e10/model.pt
  python plot_reconstructions.py --model_path ... --gaia_id 1843146113696239616
  python plot_reconstructions.py --model_path ... --tic_id 282358593 --offset 16
  python plot_reconstructions.py --model_path ... --seed 42 --offset 8

A star with multiple sectors picks one sector at random (seeded).
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lcgen.models.simple_min_gru import BiDirectionalMinGRU
from lcgen.models.TimeSeriesDataset import METADATA_FEATURES
from lcgen.models.MetadataAgePredictor import MetadataStandardizer
from lcgen.utils.mask import apply_block_mask


def generate_block_mask(L, min_size, max_size, mask_portion):
    """Block mask matching training (lcgen.utils.mask.apply_block_mask).

    Returns a (L,) float32 array with 1=observed, 0=masked.
    """
    if mask_portion <= 0:
        return np.ones(L, dtype=np.float32)
    dummy = np.zeros((1, L), dtype=np.float32)
    _, mask2d = apply_block_mask(dummy, min_size, max_size, mask_portion)
    return mask2d[0]


def mask_intervals(mask_1d):
    """Return list of (start, end) index ranges where mask_1d == 0 (masked)."""
    masked = (np.asarray(mask_1d) == 0)
    intervals = []
    in_block = False
    start = None
    for i, v in enumerate(masked):
        if v and not in_block:
            in_block = True
            start = i
        elif not v and in_block:
            in_block = False
            intervals.append((start, i))
    if in_block:
        intervals.append((start, len(masked)))
    return intervals


def load_model(model_path, device, hidden_size=64, direction='bi', mode='parallel',
               num_meta_features=0, use_conv_channels=False, conv_config=None):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

    has_meta = any(k.startswith('meta_encoder.') for k in sd)
    has_conv = any(k.startswith('ps_encoder.') for k in sd)
    has_flow = any(k.startswith('flow.') for k in sd)

    if not has_meta and num_meta_features > 0:
        print(f'[model] checkpoint has no meta_encoder; overriding num_meta_features {num_meta_features} -> 0')
        num_meta_features = 0
    if not has_conv and use_conv_channels:
        print('[model] checkpoint has no conv encoders; overriding use_conv_channels -> False')
        use_conv_channels = False
    if isinstance(ckpt, dict) and 'num_meta_features' in ckpt:
        num_meta_features = ckpt['num_meta_features']
    _d = ckpt if isinstance(ckpt, dict) else {}
    meta_use_mask = _d.get('meta_use_mask', False)

    model = BiDirectionalMinGRU(
        hidden_size=hidden_size,
        direction=direction,
        mode=mode,
        use_flow=has_flow,
        num_meta_features=num_meta_features,
        use_conv_channels=use_conv_channels,
        conv_config=conv_config,
        meta_use_mask=meta_use_mask,
        split_meta_encoders=_d.get('split_meta_encoders', False),
        instr_emb_dim=_d.get('instr_emb_dim', 16),
        adversarial_sector_head=_d.get('adversarial_sector_head', False),
        num_sectors=_d.get('num_sectors', 0),
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f'[model] loaded {model_path} (hidden={hidden_size}, direction={direction}, '
          f'meta={num_meta_features}, conv={use_conv_channels}, flow={has_flow})')
    if not has_flow:
        print('[model] WARNING: no flow head; predictions will use Gaussian head (no uncertainty band).')
    return model


def build_lightcurve_index(h5_paths):
    index = []
    for p in h5_paths:
        with h5py.File(p, 'r') as f:
            lengths = f['length'][:]
            gaia = f['metadata']['GaiaDR3_ID'][:]
            tic = f['metadata']['tic'][:]
            sector = f['metadata']['sector'][:]
        for i in range(len(lengths)):
            gid = gaia[i].decode('utf-8').strip() if isinstance(gaia[i], bytes) else str(gaia[i]).strip()
            index.append({
                'file': str(p),
                'row': int(i),
                'gaia_id': gid,
                'tic_id': int(tic[i]),
                'sector': int(sector[i]),
                'length': int(lengths[i]),
            })
    return index


def select_examples(index, gaia_id=None, tic_id=None, num_examples=3, seed=0):
    rng = np.random.default_rng(seed)
    if gaia_id is not None:
        matches = [e for e in index if e['gaia_id'] == str(gaia_id)]
        if not matches:
            raise ValueError(f'No light curve found for Gaia ID {gaia_id}')
        print(f'[select] Gaia {gaia_id}: {len(matches)} sector(s) found, picking one at random')
        return [matches[int(rng.integers(0, len(matches)))]]
    if tic_id is not None:
        matches = [e for e in index if e['tic_id'] == int(tic_id)]
        if not matches:
            raise ValueError(f'No light curve found for TIC ID {tic_id}')
        print(f'[select] TIC {tic_id}: {len(matches)} sector(s) found, picking one at random')
        return [matches[int(rng.integers(0, len(matches)))]]
    rows = rng.choice(len(index), size=num_examples, replace=False)
    return [index[int(r)] for r in rows]


def load_lightcurve(entry, use_metadata=False, use_conv_channels=False, device='cpu',
                    trim_edges=0):
    with h5py.File(entry['file'], 'r') as f:
        L = entry['length']
        row = entry['row']
        # Trim the first/last `trim_edges` raw samples so the encoder sees only
        # the interior of the light curve — matches training-time trimming.
        if trim_edges > 0:
            if L <= 2 * trim_edges:
                raise ValueError(
                    f'Light curve at row {row} (length {L}) is shorter than '
                    f'2*trim_edges={2*trim_edges}; cannot trim.'
                )
            slc_start = trim_edges
            slc_stop  = L - trim_edges
        else:
            slc_start = 0
            slc_stop  = L
        flux = f['flux'][row, slc_start:slc_stop]
        flux_err = f['flux_err'][row, slc_start:slc_stop]
        times = f['time'][row, slc_start:slc_stop]

        metadata = None
        if use_metadata and 'metadata' in f:
            raw = {}
            for feat in METADATA_FEATURES:
                if feat in f['metadata']:
                    raw[feat] = np.asarray([f['metadata'][feat][row]], dtype=np.float32)
                else:
                    raw[feat] = np.zeros(1, dtype=np.float32)
            metadata = MetadataStandardizer(fields=METADATA_FEATURES).transform(raw)[0]

    conv_data = None
    if use_conv_channels:
        stem = entry['file'].rsplit('.h5', 1)[0]
        spectra = Path(stem + '_spectra.h5')
        src = spectra if spectra.exists() else Path(entry['file'])
        with h5py.File(src, 'r') as sf:
            conv_data = {
                'power':  np.asarray(sf['power'][row],  dtype=np.float32),
                'f_stat': np.asarray(sf['f_stat'][row], dtype=np.float32),
                'acf':    np.asarray(sf['acf'][row],    dtype=np.float32),
            }

    return {
        'flux':     torch.tensor(flux,     dtype=torch.float32, device=device),
        'flux_err': torch.tensor(flux_err, dtype=torch.float32, device=device),
        'times':    torch.tensor(times,    dtype=torch.float32, device=device),
        'metadata': torch.tensor(metadata, dtype=torch.float32, device=device) if metadata is not None else None,
        'conv_data': {k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0) for k, v in conv_data.items()} if conv_data else None,
    }


def compute_flow_predictions(model, lc, offset, mask=None, n_samples=64, device='cpu'):
    """Predict flux at target positions j where j = source + k.

    Returns (pred_times, median, p16, p84). Uses the same src/target indexing as the
    bidirectional training loss, including the nearest-unmasked source remapping:
    forward src = h_fwd at nearest unmasked index <= j-k, backward src = h_bwd at
    nearest unmasked index >= j+k. If j-k (or j+k) is unmasked the remap is a no-op.

    If `mask` is given (numpy array of length L with 1=observed, 0=masked) it is
    passed to model.forward just like during training — flux/flux_err at masked
    positions are zeroed in the RNN input — and predictions are computed for every
    target position including those inside masked regions.
    """
    flux     = lc['flux'].unsqueeze(0)       # (1, L)
    flux_err = lc['flux_err'].unsqueeze(0)
    times    = lc['times'].unsqueeze(0)
    L = flux.size(1)
    if mask is None:
        mask_tensor = torch.ones(1, L, device=device)
    else:
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
    x_in = torch.stack([flux, flux_err], dim=-1)
    t_in = times.unsqueeze(-1)
    meta = lc['metadata'].unsqueeze(0) if lc['metadata'] is not None else None
    conv_data = lc['conv_data']

    with torch.no_grad():
        out = model(x_in, t_in, mask=mask_tensor, metadata=meta, conv_data=conv_data, return_states=True)
    h_fwd = out.get('h_fwd_tensor')
    h_bwd = out.get('h_bwd_tensor')
    t_enc = out['t_enc']
    Te = t_enc.size(-1)

    k = int(offset)
    if 2 * k >= L:
        raise ValueError(f'Offset {k} too large for sequence length {L} (need 2k < L)')
    n_targets = L - 2 * k
    t_tgt = t_enc[:, k:L-k, :]

    # Build source-remap tables (nearest unmasked index <= i / >= i) so that
    # targets inside a mask block pull their source hidden states from the
    # nearest real observation on each side. Matches the training-time loss.
    idx_row = torch.arange(L, device=device).unsqueeze(0)               # (1, L)
    masked_idx_fwd = torch.where(mask_tensor > 0.5, idx_row, torch.full_like(idx_row, -1))
    fwd_remap, _ = torch.cummax(masked_idx_fwd, dim=1)
    fwd_remap = torch.clamp(fwd_remap, min=0)
    masked_idx_bwd = torch.where(mask_tensor > 0.5, idx_row, torch.full_like(idx_row, L))
    bwd_flipped, _ = torch.cummin(masked_idx_bwd.flip(dims=[1]), dim=1)
    bwd_remap = bwd_flipped.flip(dims=[1])
    bwd_remap = torch.clamp(bwd_remap, max=L - 1)

    fwd_idx_slice = fwd_remap[:, :n_targets].long()
    bwd_idx_slice = bwd_remap[:, 2*k:].long()

    if model.direction == 'bi' and h_fwd is not None and h_bwd is not None:
        H = h_fwd.size(-1)
        src_f = torch.gather(h_fwd, dim=1, index=fwd_idx_slice.unsqueeze(-1).expand(-1, -1, H))
        src_b = torch.gather(h_bwd, dim=1, index=bwd_idx_slice.unsqueeze(-1).expand(-1, -1, H))
        flat_in = torch.cat([src_f, src_b, t_tgt], dim=-1).reshape(-1, 2 * H + Te)
    else:
        h = h_fwd if h_fwd is not None else h_bwd
        H = h.size(-1)
        src = torch.gather(h, dim=1, index=fwd_idx_slice.unsqueeze(-1).expand(-1, -1, H))
        flat_in = torch.cat([src, t_tgt], dim=-1).reshape(-1, H + Te)

    with torch.no_grad():
        normed = model.head_norm(flat_in)
        if Te > 0:
            normed = torch.cat([normed[:, :-Te], normed[:, -Te:] * model.time_scale], dim=1)
        ferr_flat = flux_err[:, k:L-k].reshape(-1, 1)

        if model.flow is not None:
            ctx = torch.cat([normed, ferr_flat], dim=1)
            dist = model.flow(ctx)
            samples = torch.stack([dist.sample() for _ in range(n_samples)], dim=0).squeeze(-1)
            median = torch.quantile(samples, 0.5, dim=0).cpu().numpy()
            p16    = torch.quantile(samples, 0.16, dim=0).cpu().numpy()
            p84    = torch.quantile(samples, 0.84, dim=0).cpu().numpy()
        else:
            preds = model.gauss_head(normed).squeeze(-1).cpu().numpy()
            median = preds
            p16 = preds
            p84 = preds

    pred_times = lc['times'][k:L-k].cpu().numpy()
    return pred_times, median, p16, p84


def plot_reconstructions(examples, preds_per_example, output_path, offset):
    n = len(examples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]
    for ax, entry, pr in zip(axes, examples, preds_per_example):
        ax.errorbar(pr['times'], pr['flux'], yerr=pr['flux_err'],
                    fmt='.', color='black', ecolor='lightgray',
                    markersize=2, alpha=0.6, elinewidth=0.5, label='measured')
        ax.fill_between(pr['pred_times'], pr['p16'], pr['p84'],
                        color='steelblue', alpha=0.3, linewidth=0,
                        label=f'flow p16-p84 (k={offset})')
        ax.plot(pr['pred_times'], pr['median'],
                color='steelblue', linewidth=1.2, label='flow median')
        mask = pr.get('mask')
        if mask is not None:
            first = True
            for s, e in mask_intervals(mask):
                x0 = pr['times'][s]
                x1 = pr['times'][e - 1] if e > s else x0
                ax.axvspan(x0, x1, color='red', alpha=0.12, linewidth=0,
                           label='masked' if first else None)
                first = False
        ax.set_title(f"Gaia {entry['gaia_id']} | TIC {entry['tic_id']} | sector {entry['sector']} | L={entry['length']}",
                     fontsize=11)
        ax.set_xlabel('time (BJD)')
        ax.set_ylabel('flux')
        ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[save] {output_path}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--h5_paths', type=str, nargs='+', default=[
        'final_pretrain/timeseries_pretrain.h5',
        'final_pretrain/timeseries_exop_hosts.h5',
    ])
    p.add_argument('--gaia_id', type=str, default=None, help='Gaia DR3 ID for a specific star')
    p.add_argument('--tic_id',  type=int, default=None, help='TIC ID for a specific star')
    p.add_argument('--num_examples', type=int, default=3, help='Number of random examples (when no ID given)')
    p.add_argument('--offset', type=int, default=8, help='Flow prediction offset k in timesteps')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output_dir', type=str, default='output/reconstructions')
    p.add_argument('--output_name', type=str, default=None)
    p.add_argument('--n_flow_samples', type=int, default=64)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--direction', type=str, default='bi', choices=['forward', 'backward', 'bi'])
    p.add_argument('--mode', type=str, default='parallel', choices=['sequential', 'parallel'])
    p.add_argument('--use_metadata', action='store_true')
    p.add_argument('--use_conv_channels', action='store_true')
    p.add_argument('--conv_encoder_type', type=str, default='unet', choices=['lightweight', 'unet'])
    p.add_argument('--mask_portion', type=float, default=0.0,
                   help='Fraction of timesteps to block-mask (0 = no masking). Mirrors training-time masking.')
    p.add_argument('--mask_min_size', type=int, default=2, help='Minimum mask block size')
    p.add_argument('--mask_max_size', type=int, default=40, help='Maximum mask block size')
    p.add_argument('--trim_edges', type=int, default=10,
                   help='Strip the first/last N samples from every light curve before '
                        'feeding it to the encoder. Must match the value used during '
                        'training, otherwise the model sees out-of-distribution inputs '
                        'at the sequence edges. Use 0 to disable.')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    num_meta_features = len(METADATA_FEATURES) if args.use_metadata else 0
    conv_config = None
    if args.use_conv_channels:
        conv_config = {
            'encoder_type': args.conv_encoder_type,
            'input_length': 16000,
            'encoder_dims': [4, 8, 16, 32],
            'num_layers': 4 if args.conv_encoder_type == 'unet' else 3,
            'hidden_channels': 16,
            'activation': 'gelu',
        }
    model = load_model(args.model_path, device,
                       hidden_size=args.hidden_size, direction=args.direction, mode=args.mode,
                       num_meta_features=num_meta_features,
                       use_conv_channels=args.use_conv_channels, conv_config=conv_config)

    # Auto-sync feature flags with what the checkpoint actually contains so
    # load_lightcurve loads the inputs the model expects.
    args.use_metadata = model.meta_encoder is not None
    args.use_conv_channels = bool(getattr(model, 'use_conv_channels', False))
    print(f'[auto-sync] use_metadata={args.use_metadata} use_conv_channels={args.use_conv_channels}')

    print(f'[index] scanning {len(args.h5_paths)} H5 file(s)')
    index = build_lightcurve_index(args.h5_paths)
    print(f'[index] total {len(index)} light curves available')

    examples = select_examples(index, gaia_id=args.gaia_id, tic_id=args.tic_id,
                               num_examples=args.num_examples, seed=args.seed)

    preds_per_example = []
    for ex in examples:
        eff_len = ex['length'] - 2 * args.trim_edges if args.trim_edges > 0 else ex['length']
        print(f'[lc] {ex["file"]}:row{ex["row"]} gaia={ex["gaia_id"]} tic={ex["tic_id"]} '
              f'sector={ex["sector"]} L={ex["length"]} L_after_trim={eff_len}')
        lc = load_lightcurve(ex, use_metadata=args.use_metadata,
                             use_conv_channels=args.use_conv_channels, device=device,
                             trim_edges=args.trim_edges)
        mask = generate_block_mask(eff_len, args.mask_min_size, args.mask_max_size, args.mask_portion)
        pred_times, median, p16, p84 = compute_flow_predictions(
            model, lc, offset=args.offset, mask=mask,
            n_samples=args.n_flow_samples, device=device)
        preds_per_example.append({
            'times':    lc['times'].cpu().numpy(),
            'flux':     lc['flux'].cpu().numpy(),
            'flux_err': lc['flux_err'].cpu().numpy(),
            'pred_times': pred_times,
            'median': median, 'p16': p16, 'p84': p84,
            'mask': mask,
        })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name is None:
        tag = (f'gaia{args.gaia_id}' if args.gaia_id else
               f'tic{args.tic_id}'  if args.tic_id  else
               f'seed{args.seed}')
        args.output_name = f'reconstructions_{tag}_k{args.offset}.png'
    plot_reconstructions(examples, preds_per_example, out_dir / args.output_name, offset=args.offset)


if __name__ == '__main__':
    main()
