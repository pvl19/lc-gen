"""Per-timestep saliency for a single light curve.

Three measures, all written to one .npz per star (see --output_dir):

(1) Integrated Gradients of f(z) = ||z||^2 w.r.t. flux and flux_err at each
    timestep, where z is the pooled multiscale latent (the vector the age
    inference pipeline consumes). Saturation-aware and pool-aware.
    Keys: s_flux_norm, s_err_norm (both (L,)).
(2) Integrated Gradients of f(z) = (z - z_mean) @ pc1, where pc1 is the
    leading PC of a latent bank supplied via --latents_npz. PC1 picks out
    the most age-relevant direction the downstream pipeline actually uses.
    Optional; skipped silently if --latents_npz is not given.
    Keys: s_flux_pc1, s_err_pc1.
(3) Analytic minGRU update-gate memory weight, bidirectional. For each step
    t, w_t = avg_dim avg_T [ z_t * prod_{s in (t,T]}(1 - z_s) ] over future
    indices T. Encoder-side only — does NOT include the pool.
    Key: w_gate (L,).

The output .npz also stores: times, flux, flux_err, mask, f_norm_full,
f_norm_baseline, f_pc1_full, f_pc1_baseline (the latter two only if PCA used).
Completeness checks against these are run inside this script and printed; the
sanity panel in plot_saliency.py re-prints them.

Usage (called from saliency.sh; all params hardcoded in that wrapper):
  python scripts/compute_saliency.py --model_path ... --gaia_id ... --output_dir ...
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lcgen.models.simple_min_gru import BiDirectionalMinGRU  # noqa: F401  (used via plot_reconstructions.load_model)
from lcgen.models.TimeSeriesDataset import METADATA_FEATURES

# Reuse the proven loaders from the reconstruction script.
from plot_reconstructions import (
    load_model,
    build_lightcurve_index,
    select_examples,
    load_lightcurve,
)
# And the exact pool that the age-inference latents go through.
from plot_umap_latent import compute_multiscale_features


# ---------------------------------------------------------------------------
# Core: run model forward and produce the pooled latent z, fully differentiable
# w.r.t. (flux, flux_err) so we can backprop IG steps through it.
# ---------------------------------------------------------------------------

def _pooled_latent(model, x_in, t_in, mask, meta, meta_mask, conv_data,
                   apply_head_norm: bool, pool_kwargs: dict):
    """Run model -> hidden states -> head_norm -> multiscale pool -> z.

    Mirrors the path inside extract_latent_vectors in plot_umap_latent.py so
    that the saliency targets the EXACT vector downstream age inference uses.
    x_in is (1, L, 2); the returned z is (D,) where D = 12 * 2H.
    """
    out = model(x_in, t_in, mask=mask, metadata=meta, meta_mask=meta_mask,
                conv_data=conv_data, return_states=True)
    h_fwd = out.get('h_fwd_tensor')   # (1, L, H)
    h_bwd = out.get('h_bwd_tensor')   # (1, L, H)
    t_enc = out.get('t_enc')          # (1, L, Te)
    if h_fwd is None or h_bwd is None:
        raise RuntimeError(
            'Saliency requires a bi-directional model returning both '
            'h_fwd_tensor and h_bwd_tensor.'
        )

    if apply_head_norm:
        if model.head_norm is None:
            raise RuntimeError(
                '--apply_head_norm set but model has no head_norm. Pass '
                '--no-apply_head_norm or use a checkpoint with head_norm.'
            )
        H = model.hidden_size
        h_bi = torch.cat([h_fwd, h_bwd, t_enc], dim=-1)
        h_bi = model.head_norm(h_bi)
        h_fwd = h_bi[..., :H]
        h_bwd = h_bi[..., H:2 * H]

    # (1, L, 2H) -> (L, 2H). mask is all-ones at inference (we pass no test-time
    # masking), so valid_len == L.
    L = x_in.size(1)
    h_combined = torch.cat([h_fwd[0, :L, :], h_bwd[0, :L, :]], dim=-1)
    t_combined = t_in[0, :L] if t_in.dim() == 2 else t_in[0, :L, 0]

    z = compute_multiscale_features(
        h_combined, t_combined,
        n_segments=4,
        hidden_size=model.hidden_size,
        **pool_kwargs,
    )
    return z


def _build_x_in(flux: torch.Tensor, flux_err: torch.Tensor):
    """Stack into (1, L, 2) on flux.device and mark for autograd."""
    x = torch.stack([flux, flux_err], dim=-1).unsqueeze(0).clone()
    x.requires_grad_(True)
    return x


# ---------------------------------------------------------------------------
# (1, 2) Integrated Gradients
# ---------------------------------------------------------------------------

def integrated_gradients(model, lc, baseline_mode: str, n_steps: int,
                         target: str, pca_dir=None, pca_mean=None,
                         apply_head_norm: bool = True,
                         pool_kwargs=None, device='cpu', verbose=True):
    """Integrated Gradients of f(z) w.r.t. flux and flux_err.

    Args:
        model: BiDirectionalMinGRU (eval mode).
        lc: dict from load_lightcurve (flux, flux_err, times, metadata, conv_data).
        baseline_mode: 'zero' (zero flux, zero flux_err) or 'mean' (per-sequence
            mean flux, median flux_err).
        n_steps: number of trapezoidal-rule IG steps (>= 16 recommended).
        target: 'norm' (f = ||z||^2) or 'pc1' (f = (z - pca_mean) @ pca_dir).
        pca_dir, pca_mean: required for 'pc1'.
        apply_head_norm: feed pooled hidden states through head_norm (matches
            extract_latent_vectors used to build the downstream latent bank).
        pool_kwargs: forwarded to compute_multiscale_features. Should match
            the bank's extraction config.

    Returns:
        dict with:
            s_flux: (L,) attribution per timestep on flux channel.
            s_err:  (L,) attribution per timestep on flux_err channel.
            f_full: scalar f(z) at x (no path scaling).
            f_baseline: scalar f(z') at the baseline.
        Completeness: sum(s_flux + s_err) ≈ f_full - f_baseline.
    """
    pool_kwargs = pool_kwargs or {}

    flux = lc['flux'].to(device)
    flux_err = lc['flux_err'].to(device)
    times = lc['times'].to(device)
    L = flux.numel()

    # Baseline input.
    if baseline_mode == 'zero':
        flux_b = torch.zeros_like(flux)
        err_b = torch.zeros_like(flux_err)
    elif baseline_mode == 'mean':
        flux_b = torch.full_like(flux, float(flux.mean()))
        err_b = torch.full_like(flux_err, float(flux_err.median()))
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode!r}")

    # Static inputs to the model.
    t_in = times.unsqueeze(0).unsqueeze(-1)        # (1, L, 1)
    mask = torch.ones(1, L, device=device)
    meta = lc['metadata'].unsqueeze(0) if lc['metadata'] is not None else None
    if meta is not None and model.meta_use_mask:
        meta_mask = torch.ones_like(meta)
    else:
        meta_mask = None
    conv_data = lc['conv_data']

    def f_of_z(z):
        if target == 'norm':
            return (z * z).sum()
        elif target == 'pc1':
            return ((z - pca_mean) * pca_dir).sum()
        else:
            raise ValueError(f"Unknown target {target!r}")

    # Trapezoidal IG: alphas at midpoints of N equal intervals.
    # IG_t ≈ (x_t - x'_t) * (1/N) * Σ_{k=0..N-1} ∂f/∂x at alpha=(k+0.5)/N.
    alphas = (torch.arange(n_steps, device=device, dtype=flux.dtype) + 0.5) / n_steps

    s_flux_acc = torch.zeros_like(flux)
    s_err_acc = torch.zeros_like(flux_err)

    for k, alpha in enumerate(alphas):
        flux_a = flux_b + alpha * (flux - flux_b)
        err_a = err_b + alpha * (flux_err - err_b)
        x_a = torch.stack([flux_a, err_a], dim=-1).unsqueeze(0).clone()
        x_a.requires_grad_(True)

        z = _pooled_latent(
            model, x_a, t_in, mask, meta, meta_mask, conv_data,
            apply_head_norm=apply_head_norm, pool_kwargs=pool_kwargs,
        )
        scalar = f_of_z(z)
        grad, = torch.autograd.grad(scalar, x_a, retain_graph=False, create_graph=False)
        # grad: (1, L, 2). Average gradient over k contributes to IG.
        s_flux_acc = s_flux_acc + grad[0, :, 0].detach()
        s_err_acc = s_err_acc + grad[0, :, 1].detach()
        if verbose and (k + 1) % max(1, n_steps // 4) == 0:
            print(f'  [IG/{target}] step {k+1}/{n_steps}')

    s_flux = (flux - flux_b) * (s_flux_acc / n_steps)
    s_err = (flux_err - err_b) * (s_err_acc / n_steps)

    # Endpoint values for completeness check.
    with torch.no_grad():
        x_full = torch.stack([flux, flux_err], dim=-1).unsqueeze(0)
        z_full = _pooled_latent(model, x_full, t_in, mask, meta, meta_mask, conv_data,
                                apply_head_norm=apply_head_norm, pool_kwargs=pool_kwargs)
        f_full = f_of_z(z_full).item()
        x_base = torch.stack([flux_b, err_b], dim=-1).unsqueeze(0)
        z_base = _pooled_latent(model, x_base, t_in, mask, meta, meta_mask, conv_data,
                                apply_head_norm=apply_head_norm, pool_kwargs=pool_kwargs)
        f_baseline = f_of_z(z_base).item()

    return {
        's_flux': s_flux.detach().cpu().numpy(),
        's_err': s_err.detach().cpu().numpy(),
        'f_full': f_full,
        'f_baseline': f_baseline,
        'z_full': z_full.detach().cpu().numpy(),
        'z_baseline': z_base.detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# (3) Analytic minGRU update-gate memory weighting (Method 4 in the plan)
# ---------------------------------------------------------------------------

def _capture_gate_logits(model):
    """Return (handles, captured_dict). Hooks W_z of both encoder cells so we
    can read the pre-sigmoid gate logits without re-deriving the projection.
    """
    captured = {}

    def make_hook(name):
        def hook(_module, _inp, out):
            captured[name] = out.detach()
        return hook

    handles = []
    if hasattr(model, 'forward_cell'):
        handles.append(model.forward_cell.W_z.register_forward_hook(make_hook('k_fwd')))
    if hasattr(model, 'backward_cell'):
        handles.append(model.backward_cell.W_z.register_forward_hook(make_hook('k_bwd')))
    return handles, captured


def _gate_weight_one_direction(k_logits: torch.Tensor):
    """For one direction, return two complementary per-step weights, both
    averaged over hidden dims.

    k_logits: (1, L, H) pre-sigmoid logits captured from W_z. The minGRU
    update is h_t = (1 - z_t) h_{t-1} + z_t tilde_h_t with z_t = sigmoid(k_t).

    Returns (w_write, w_final), each shape (L,):
      w_write  = mean_d z_{t,d} — how strongly this step writes to memory.
                 Non-decaying; useful for the headline visualization since
                 long-horizon decay does NOT wash out interesting local
                 structure. This is the most direct "memory write
                 intensity" trace.
      w_final  = mean_d [z_{t,d} * prod_{s=t+1..L-1} (1 - z_{s,d})] — the
                 fraction of step t's write that survives to the final
                 hidden state. Decays geometrically so most of a long
                 sequence saturates near zero — informative for memory
                 horizon but not for visualization. Saved for diagnostics.
    """
    k = k_logits[0]            # (L, H)
    L, H = k.shape
    z = torch.sigmoid(k)
    w_write = z.mean(dim=1)

    log_z = torch.nn.functional.logsigmoid(k)
    log_om = torch.nn.functional.logsigmoid(-k)
    C = torch.cumsum(log_om, dim=0)
    log_w = log_z + (C[-1:, :] - C)
    w_final = log_w.exp().mean(dim=1)
    return w_write.detach(), w_final.detach()


def gate_weight_attribution(model, lc, apply_head_norm: bool,
                            meta_use_mask: bool, device='cpu'):
    """Run one forward pass with W_z hooks and return the bidirectional
    per-step memory weight w_t (L,).
    """
    flux = lc['flux'].to(device)
    flux_err = lc['flux_err'].to(device)
    times = lc['times'].to(device)
    L = flux.numel()
    x_in = torch.stack([flux, flux_err], dim=-1).unsqueeze(0)
    t_in = times.unsqueeze(0).unsqueeze(-1)
    mask = torch.ones(1, L, device=device)
    meta = lc['metadata'].unsqueeze(0) if lc['metadata'] is not None else None
    meta_mask = torch.ones_like(meta) if (meta is not None and meta_use_mask) else None

    handles, captured = _capture_gate_logits(model)
    try:
        with torch.no_grad():
            _ = model(x_in, t_in, mask=mask, metadata=meta, meta_mask=meta_mask,
                      conv_data=lc['conv_data'], return_states=True)
    finally:
        for h in handles:
            h.remove()

    if 'k_fwd' not in captured or 'k_bwd' not in captured:
        raise RuntimeError(
            'Could not capture W_z logits — model is not bi-directional or '
            'forward/backward cells are named differently.'
        )

    # Forward direction in original time order.
    w_write_fwd, w_final_fwd = _gate_weight_one_direction(captured['k_fwd'])

    # Backward direction: the parallel scan flips the sequence, then
    # h_bwd_tensor is unflipped after. The captured k_bwd is in REVERSED time
    # order, so we flip the per-step traces back to original time.
    w_write_bwd, w_final_bwd = _gate_weight_one_direction(captured['k_bwd'])
    w_write_bwd = torch.flip(w_write_bwd, dims=[0])
    w_final_bwd = torch.flip(w_final_bwd, dims=[0])

    return {
        'w_write': (0.5 * (w_write_fwd + w_write_bwd)).cpu().numpy(),
        'w_final': (0.5 * (w_final_fwd + w_final_bwd)).cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _load_latent_bank(path: str) -> np.ndarray:
    """Load one latents .npz and return its (N, D) latent matrix."""
    data = np.load(path, allow_pickle=True)
    if 'latent_vectors' in data:
        X = data['latent_vectors']
    elif 'latents' in data:
        X = data['latents']
    else:
        keys = [k for k in data.files if data[k].ndim == 2]
        if not keys:
            raise ValueError(f'No 2D array found in {path}; keys: {list(data.files)}')
        X = data[keys[0]]
        print(f'[pca]   {path}: using array key {keys[0]!r}')
    return np.asarray(X)


def maybe_load_pc1(latents_npz, device):
    """Load one or more latents .npz files, concatenate row-wise, fit PCA, and
    return (pc1_dir, mean) on `device`, where pc1_dir is expressed in the
    ORIGINAL latent coordinates so f(z) = (z - mean) @ pc1_dir.

    Pre-PCA: per-feature z-scoring (X - X_mean) / X_std, matching the
    deployment PCA in kfold_age_inference.fit_global_pca_bundle so PC1 is the
    direction the age head would actually pick up. Without z-scoring, the
    1536-d multiscale latent's natural-scale heterogeneity across pool blocks
    (order stats dominate variance) lets one block monopolize PC1 — what the
    saliency would attribute to PC1 then has nothing to do with the
    age-inference projection.

    latents_npz: str | list[str] | None.
    All banks must share the same D (pool dim).
    """
    if not latents_npz:
        return None, None
    paths = [latents_npz] if isinstance(latents_npz, str) else list(latents_npz)
    parts = []
    print(f'[pca] loading {len(paths)} latent bank(s)')
    for p in paths:
        X = _load_latent_bank(p)
        print(f'[pca]   {p}: shape {X.shape}')
        parts.append(X)
    Ds = {p.shape[1] for p in parts}
    if len(Ds) != 1:
        raise ValueError(f'Latent banks have mismatched feature dims: {Ds}. '
                         f'They must come from the same checkpoint + extraction config.')
    X = np.concatenate(parts, axis=0).astype(np.float64, copy=False)
    print(f'[pca] fitting PCA on concatenated bank of shape {X.shape} '
          f'(per-feature z-score, matches deployment)')
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    cov = (X_norm.T @ X_norm) / max(1, X_norm.shape[0] - 1)
    w, v = np.linalg.eigh(cov)
    pc1_norm = v[:, -1]
    pc1_norm = pc1_norm / max(np.linalg.norm(pc1_norm), 1e-12)
    if (X_norm @ pc1_norm).mean() < 0:
        pc1_norm = -pc1_norm
    # Re-express PC1 in the ORIGINAL coordinates so we can compute the target
    # scalar as (z_raw - X_mean) @ pc1_orig without re-standardizing inside the
    # IG loop. From (z - X_mean) @ pc1_orig := ((z - X_mean) / X_std) @ pc1_norm,
    # we have pc1_orig = pc1_norm / X_std. The result is not unit-norm in the
    # original space — that's fine; only directionality matters for IG.
    pc1_orig = pc1_norm / X_std
    print(f'[pca] D={pc1_orig.size}, eig_top/eig_sum = {w[-1] / max(w.sum(), 1e-12):.3f} '
          f'(on z-scored bank)')
    return (
        torch.tensor(pc1_orig, dtype=torch.float32, device=device),
        torch.tensor(X_mean, dtype=torch.float32, device=device),
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--h5_paths', type=str, nargs='+', required=True,
                   help='H5 files containing candidate light curves.')
    p.add_argument('--gaia_id', type=str, default=None)
    p.add_argument('--tic_id', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output_dir', type=str, default='output/saliency')
    p.add_argument('--n_ig_steps', type=int, default=64,
                   help='Number of trapezoidal IG steps. 32-64 is standard.')
    p.add_argument('--baseline_mode', type=str, default='zero', choices=['zero', 'mean'])
    p.add_argument('--latents_npz', type=str, nargs='+', default=None,
                   help='Optional path(s) to one or more latent-bank .npz files. '
                        'When multiple paths are given they are concatenated '
                        'row-wise before PCA — gives a PC1 representative of '
                        'the full deployment population (pretrain + hosts + '
                        'thickdisk) rather than the labeled subset alone. If '
                        'given, IG is also computed for the PC1 target.')
    p.add_argument('--trim_edges', type=int, default=10,
                   help='Must match the model training. Same convention as in '
                        'plot_reconstructions / plot_umap_latent.')
    p.add_argument('--apply_head_norm', action=argparse.BooleanOptionalAction, default=True,
                   help='Match the latent-bank extraction (default True).')
    p.add_argument('--minmax_edge_skip', type=int, default=100,
                   help='Match plot_umap_metaAll.sh (default 100).')
    p.add_argument('--minmax_quantile', type=float, default=0.0)
    p.add_argument('--subtract_temporal_mean', action='store_true')
    p.add_argument('--glob_mode', type=str, default='uniform')
    p.add_argument('--seg_mode', type=str, default='equal_count')
    p.add_argument('--diff_weight_mode', type=str, default='dt')
    # Model-build knobs (match the checkpoint)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--direction', type=str, default='bi', choices=['forward', 'backward', 'bi'])
    p.add_argument('--mode', type=str, default='parallel', choices=['sequential', 'parallel'])
    p.add_argument('--use_metadata', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--use_conv_channels', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    num_meta = len(METADATA_FEATURES) if args.use_metadata else 0
    model = load_model(args.model_path, device,
                       hidden_size=args.hidden_size,
                       direction=args.direction,
                       mode=args.mode,
                       num_meta_features=num_meta,
                       use_conv_channels=args.use_conv_channels,
                       conv_config=None)
    args.use_metadata = model.meta_encoder is not None
    print(f'[auto-sync] use_metadata={args.use_metadata}')

    pca_dir, pca_mean = maybe_load_pc1(args.latents_npz, device)

    index = build_lightcurve_index(args.h5_paths)
    print(f'[index] {len(index)} light curves available')
    entries = select_examples(index, gaia_id=args.gaia_id, tic_id=args.tic_id,
                              num_examples=1, seed=args.seed)

    pool_kwargs = dict(
        minmax_edge_skip=args.minmax_edge_skip,
        minmax_quantile=args.minmax_quantile,
        subtract_temporal_mean=args.subtract_temporal_mean,
        glob_mode=args.glob_mode,
        seg_mode=args.seg_mode,
        diff_weight_mode=args.diff_weight_mode,
    )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for ex in entries:
        tag = f'gaia{ex["gaia_id"]}_s{ex["sector"]}'
        print(f'\n=== {tag} | file={ex["file"]}:{ex["row"]} L={ex["length"]} ===')
        lc = load_lightcurve(ex, use_metadata=args.use_metadata,
                             use_conv_channels=args.use_conv_channels,
                             device=device, trim_edges=args.trim_edges)
        L = lc['flux'].numel()

        # --- (1) IG: ||z||^2 target ---
        print(f'[IG] target=||z||^2, n_steps={args.n_ig_steps}, baseline={args.baseline_mode}')
        ig_norm = integrated_gradients(
            model, lc, baseline_mode=args.baseline_mode, n_steps=args.n_ig_steps,
            target='norm', apply_head_norm=args.apply_head_norm,
            pool_kwargs=pool_kwargs, device=device,
        )

        # --- (2) IG: PC1 target (optional) ---
        ig_pc1 = None
        if pca_dir is not None:
            print(f'[IG] target=z@pc1, n_steps={args.n_ig_steps}, baseline={args.baseline_mode}')
            ig_pc1 = integrated_gradients(
                model, lc, baseline_mode=args.baseline_mode, n_steps=args.n_ig_steps,
                target='pc1', pca_dir=pca_dir, pca_mean=pca_mean,
                apply_head_norm=args.apply_head_norm,
                pool_kwargs=pool_kwargs, device=device,
            )

        # --- (3) Gate-weight ---
        print('[gate] computing analytic memory weights')
        gate = gate_weight_attribution(
            model, lc, apply_head_norm=args.apply_head_norm,
            meta_use_mask=model.meta_use_mask, device=device,
        )
        w_gate = gate['w_write']
        w_gate_final = gate['w_final']

        # --- Sanity checks ---
        sanity = {}
        for name, ig in [('norm', ig_norm)] + ([('pc1', ig_pc1)] if ig_pc1 is not None else []):
            total = float(np.sum(ig['s_flux']) + np.sum(ig['s_err']))
            expected = ig['f_full'] - ig['f_baseline']
            rel = abs(total - expected) / max(abs(expected), 1e-12)
            sanity[f'{name}_completeness_total'] = total
            sanity[f'{name}_completeness_expected'] = expected
            sanity[f'{name}_completeness_rel'] = rel
            ok = rel < 1e-2
            print(f'[sanity:{name}] sum(s_flux+s_err)={total:.6e}  f-f0={expected:.6e}  '
                  f'rel_err={rel:.3e}  {"OK" if ok else "WARN"}')

        # Per-attribution shape check
        assert ig_norm['s_flux'].shape == (L,)
        assert ig_norm['s_err'].shape == (L,)
        assert w_gate.shape == (L,)

        # --- Save ---
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / 'attribution.npz'
        save_dict = dict(
            gaia_id=ex['gaia_id'], tic_id=ex['tic_id'], sector=ex['sector'],
            length=L, trim_edges=args.trim_edges,
            baseline_mode=args.baseline_mode, n_ig_steps=args.n_ig_steps,
            times=lc['times'].detach().cpu().numpy(),
            flux=lc['flux'].detach().cpu().numpy(),
            flux_err=lc['flux_err'].detach().cpu().numpy(),
            mask=np.ones(L, dtype=np.float32),
            s_flux_norm=ig_norm['s_flux'],
            s_err_norm=ig_norm['s_err'],
            f_norm_full=ig_norm['f_full'],
            f_norm_baseline=ig_norm['f_baseline'],
            w_gate=w_gate,
            w_gate_final=w_gate_final,
        )
        if ig_pc1 is not None:
            save_dict.update(
                s_flux_pc1=ig_pc1['s_flux'],
                s_err_pc1=ig_pc1['s_err'],
                f_pc1_full=ig_pc1['f_full'],
                f_pc1_baseline=ig_pc1['f_baseline'],
            )
        np.savez(save_path, **save_dict)
        with open(out_dir / 'sanity.json', 'w') as f:
            json.dump(sanity, f, indent=2)
        print(f'[save] {save_path}')


if __name__ == '__main__':
    main()
