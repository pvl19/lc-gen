"""Per-direction prediction NLL: are forward hidden states more useful for
predicting flux than backward hidden states?

Diagnostic test for the bidirectional-saliency finding that forward attribution
is ~7× larger than backward attribution. Two interpretations of that gap were
on the table:

  (A) Forward channel is genuinely more predictive of flux than backward.
  (B) Both channels are equally predictive; the attribution gap is a property
      of representational sensitivity (the forward channel was just trained
      to be more responsive to flux changes), not predictive value.

This script distinguishes them by running the deployed reconstruction head
under three context-ablation modes and reporting the resulting NLL on a
sample of eval-set light curves:

  both: head consumes [h_fwd[j-k] | h_bwd[j+k] | t_enc[j]] (deployment).
  fwd:  head consumes [h_fwd[j-k] | 0           | t_enc[j]].
  bwd:  head consumes [0          | h_bwd[j+k]  | t_enc[j]].

NLL is computed by the flow head (matches the training loss
bounded_horizon_future_nll for the deployed sendit/e100 checkpoint), at a
single fixed horizon k.

CAVEAT: the head_norm and the flow were trained jointly on the full
bidirectional context. Zeroing one direction post-hoc produces an
out-of-distribution input. The resulting NLL therefore measures "how much
worse does the *deployed* pipeline get when forced to ignore one direction,"
not "how well could that direction predict if trained alone." A clean answer
to the latter would require retraining a flow conditioned on just one
direction; we ship the former as a fast diagnostic.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_reconstructions import (
    load_model, build_lightcurve_index, select_examples, load_lightcurve,
)
from lcgen.models.TimeSeriesDataset import METADATA_FEATURES


def per_direction_nll_one_star(model, lc, k: int, device: str = 'cpu'):
    """For one light curve, compute mean per-prediction NLL under three
    direction-ablation modes. Mirrors bounded_horizon_future_nll exactly
    except for the optional zeroing of h_fwd or h_bwd.
    """
    flux = lc['flux'].to(device)
    flux_err = lc['flux_err'].to(device)
    times = lc['times'].to(device)
    L = flux.numel()
    if 2 * k >= L:
        raise ValueError(f'k={k} too large for L={L}')

    x = torch.stack([flux, flux_err], dim=-1).unsqueeze(0)
    t = times.unsqueeze(0).unsqueeze(-1)
    mask = torch.ones(1, L, device=device)
    meta = lc['metadata'].unsqueeze(0) if lc['metadata'] is not None else None
    meta_mask = (torch.ones_like(meta)
                 if (meta is not None and model.meta_use_mask) else None)

    with torch.no_grad():
        out = model(x, t, mask=mask, metadata=meta, meta_mask=meta_mask,
                    return_states=True)
    h_fwd = out['h_fwd_tensor']                 # (1, L, H)
    h_bwd = out['h_bwd_tensor']                 # (1, L, H)
    t_enc = out['t_enc']                        # (1, L, Te)
    H = model.hidden_size
    Te = t_enc.size(-1)

    # Target positions j ∈ [k, L-k); forward src = j-k, backward src = j+k.
    n_targets = L - 2 * k
    t_tgt = t_enc[:, k:L - k, :]                # (1, n_targets, Te)
    src_f_full = h_fwd[:, :n_targets, :]        # (1, n_targets, H)
    src_b_full = h_bwd[:, 2 * k:, :]            # (1, n_targets, H)
    flux_tgt = flux[k:L - k]
    ferr_tgt = flux_err[k:L - k]

    results = {}
    for mode in ['both', 'fwd', 'bwd']:
        sf = src_f_full if mode != 'bwd' else torch.zeros_like(src_f_full)
        sb = src_b_full if mode != 'fwd' else torch.zeros_like(src_b_full)
        inputs_k = torch.cat([sf, sb, t_tgt], dim=-1)         # (1, n_targets, 2H+Te)
        flat_in = inputs_k.reshape(-1, 2 * H + Te)

        # head_norm + time-scale (replicates loss.py:173-180).
        normed = model.head_norm(flat_in)
        if Te > 0 and getattr(model, 'time_scale', None) is not None:
            h_hidden = normed[:, :-Te]
            h_time = normed[:, -Te:] * model.time_scale
            normed = torch.cat([h_hidden, h_time], dim=1)

        # Flow context: + flux_err at target position.
        ferr_flat = ferr_tgt.contiguous().view(-1).unsqueeze(1)
        ctx = torch.cat([normed, ferr_flat], dim=1)
        with torch.no_grad():
            dist = model.flow(ctx)
            logp = dist.log_prob(flux_tgt.contiguous().view(-1, 1))
        if logp.dim() > 1:
            logp = logp.view(-1)
        nll = -logp.mean().item()
        results[mode] = nll
    return results, n_targets


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--h5_paths', type=str, nargs='+', required=True)
    p.add_argument('--n_stars', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--k', type=int, default=8,
                   help='Fixed horizon for the prediction. 8 matches the '
                        'default offset in plot_reconstructions.sh; a few '
                        'values would let us look at horizon sensitivity.')
    p.add_argument('--ks', type=int, nargs='+', default=None,
                   help='If set, sweep these k values instead of using --k.')
    p.add_argument('--trim_edges', type=int, default=10)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--use_metadata', action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    num_meta = len(METADATA_FEATURES) if args.use_metadata else 0
    model = load_model(args.model_path, device, hidden_size=args.hidden_size,
                       num_meta_features=num_meta)
    args.use_metadata = model.meta_encoder is not None

    print(f'[index] scanning {len(args.h5_paths)} H5 file(s)')
    index = build_lightcurve_index(args.h5_paths)
    print(f'[index] {len(index)} light curves available')

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(index), size=min(args.n_stars, len(index)), replace=False)
    entries = [index[int(i)] for i in picks]

    ks = args.ks if args.ks else [args.k]
    print(f'[run] {len(entries)} stars, horizons k = {ks}')

    # rows: (star_idx, k, mode) -> nll
    all_rows = []
    for star_idx, ex in enumerate(entries):
        try:
            lc = load_lightcurve(ex, use_metadata=args.use_metadata,
                                 device=device, trim_edges=args.trim_edges)
        except Exception as e:
            print(f'  star {star_idx} (gaia {ex["gaia_id"]}): skipped — {e}')
            continue
        for k in ks:
            try:
                res, n_targets = per_direction_nll_one_star(model, lc, k=k,
                                                            device=device)
            except ValueError as e:
                print(f'  star {star_idx} k={k}: skipped — {e}')
                continue
            for mode, nll in res.items():
                all_rows.append((star_idx, k, mode, nll, n_targets))
        if (star_idx + 1) % 5 == 0:
            print(f'  done {star_idx + 1}/{len(entries)}')

    if not all_rows:
        raise SystemExit('No successful evaluations.')

    print()
    print('Per-direction NLL (lower = better predictor).')
    print('  both = deployed bidirectional context.')
    print('  fwd  = backward channel zeroed in head context.')
    print('  bwd  = forward channel zeroed in head context.')
    print()
    arr = np.array([(s, k, m, n) for (s, k, m, n, _) in all_rows],
                   dtype=[('star', int), ('k', int), ('mode', 'U8'), ('nll', float)])
    for k in ks:
        sub = arr[arr['k'] == k]
        n_stars = len(np.unique(sub['star']))
        print(f'k = {k}  (N = {n_stars} stars):')
        for mode in ['both', 'fwd', 'bwd']:
            v = sub[sub['mode'] == mode]['nll']
            if len(v) == 0:
                continue
            print(f'  {mode:4s}: mean={v.mean():.4f}  median={np.median(v):.4f}  '
                  f'std={v.std():.4f}  n={len(v)}')
        # Pairwise delta vs deployed.
        both = sub[sub['mode'] == 'both']['nll']
        fwd = sub[sub['mode'] == 'fwd']['nll']
        bwd = sub[sub['mode'] == 'bwd']['nll']
        if len(both) == len(fwd) and len(both) == len(bwd):
            d_fwd = fwd - both  # NLL increase when backward is zeroed
            d_bwd = bwd - both  # NLL increase when forward is zeroed
            print(f'  ΔNLL when bwd zeroed (fwd-only minus both): mean={d_fwd.mean():+.4f}  '
                  f'median={np.median(d_fwd):+.4f}')
            print(f'  ΔNLL when fwd zeroed (bwd-only minus both): mean={d_bwd.mean():+.4f}  '
                  f'median={np.median(d_bwd):+.4f}')
            ratio = d_bwd.mean() / max(abs(d_fwd.mean()), 1e-12)
            print(f'  Ratio Δbwd/Δfwd: {ratio:+.2f}  '
                  f'(> 1 → forward is more predictive than backward)')


if __name__ == '__main__':
    main()
