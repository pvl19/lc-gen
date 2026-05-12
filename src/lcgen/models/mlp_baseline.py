"""Local-window MLP baseline for the flux-prediction comparison.

A simple MLP that predicts flux[j] from a fixed-size window of points on
either side of the masked-out gap [j-k, j+k]. The encoder is shared; two
output heads are provided:

  - MLPGaussianBaseline: outputs (mu, log_var), trained with Gaussian NLL.
  - MLPNSFBaseline:      outputs a zuko NSF over flux, conditioned on the
                         MLP context vector. (Plumbed but not exercised in v1.)

Input layout per prediction (all 1D, concatenated):
    forward flux       [j-k-C : j-k]               length C
    backward flux      [j+k   : j+k+C]             length C
    flux_err           (forward + backward, 2C)    length 2C
    relative times     (t - t_j) for those 2C pts  length 2C
    valid mask         (1 for real, 0 for padded)  length 2C
    target's flux_err[j]                            length 1
    log2(k)                                         length 1
    metadata vector                                 length M

Total input_dim = 8*C + 2 + M.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import zuko
except Exception:  # pragma: no cover - zuko is optional
    zuko = None


class MLPBaselineEncoder(nn.Module):
    """Shared encoder: fixed-length window -> D-dim context vector."""

    def __init__(self, C: int = 32, num_meta_features: int = 13,
                 hidden_dims: tuple = (128, 128), context_dim: int = 136):
        super().__init__()
        self.C = C
        self.num_meta_features = num_meta_features
        self.context_dim = context_dim
        self.input_dim = 8 * C + 2 + num_meta_features

        layers = []
        prev = self.input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        layers += [nn.Linear(prev, context_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPGaussianBaseline(nn.Module):
    """Encoder + Gaussian head. Outputs (mu, log_var)."""

    def __init__(self, C: int = 32, num_meta_features: int = 13,
                 hidden_dims: tuple = (128, 128), context_dim: int = 136):
        super().__init__()
        self.encoder = MLPBaselineEncoder(C=C, num_meta_features=num_meta_features,
                                          hidden_dims=hidden_dims, context_dim=context_dim)
        self.head = nn.Linear(context_dim, 2)
        self.C = C
        self.num_meta_features = num_meta_features
        self.context_dim = context_dim

    def forward(self, x: torch.Tensor):
        ctx = self.encoder(x)
        out = self.head(ctx)
        mu = out[..., 0]
        log_var = out[..., 1]
        return mu, log_var


class MLPNSFBaseline(nn.Module):
    """Encoder + zuko NSF head conditioned on (context, target_ferr).

    Plumbed for v2. Forward returns log_prob(target) for training; sample()
    for eval quantiles.
    """

    def __init__(self, C: int = 32, num_meta_features: int = 13,
                 hidden_dims: tuple = (128, 128), context_dim: int = 136,
                 nsf_transforms: int = 2, nsf_hidden: tuple = (64, 64)):
        super().__init__()
        if zuko is None:
            raise RuntimeError("zuko is required for MLPNSFBaseline")
        self.encoder = MLPBaselineEncoder(C=C, num_meta_features=num_meta_features,
                                          hidden_dims=hidden_dims, context_dim=context_dim)
        # Flow conditions on context_dim + 1 (target's flux_err) — mirrors the
        # RNN's flow head conditioning shape.
        self.flow = zuko.flows.NSF(1, context_dim + 1, transforms=nsf_transforms,
                                   hidden_features=list(nsf_hidden))
        self.C = C
        self.num_meta_features = num_meta_features
        self.context_dim = context_dim

    def context(self, x: torch.Tensor, target_ferr: torch.Tensor) -> torch.Tensor:
        ctx = self.encoder(x)
        return torch.cat([ctx, target_ferr.unsqueeze(-1)], dim=-1)

    def log_prob(self, x: torch.Tensor, target_ferr: torch.Tensor,
                 target_flux: torch.Tensor) -> torch.Tensor:
        ctx = self.context(x, target_ferr)
        dist = self.flow(ctx)
        return dist.log_prob(target_flux.unsqueeze(-1)).view(-1)

    def sample(self, x: torch.Tensor, target_ferr: torch.Tensor,
               n_samples: int = 128) -> torch.Tensor:
        ctx = self.context(x, target_ferr)
        dist = self.flow(ctx)
        s = torch.stack([dist.sample() for _ in range(n_samples)], dim=0)
        return s.squeeze(-1)  # (n_samples, N)


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor,
                 clamp_logvar: tuple = (-10.0, 6.0)) -> torch.Tensor:
    """Element-wise Gaussian NLL. Clamps log_var to keep training stable."""
    log_var = log_var.clamp(min=clamp_logvar[0], max=clamp_logvar[1])
    var = torch.exp(log_var)
    return 0.5 * (log_var + (target - mu) ** 2 / var + math.log(2.0 * math.pi))


def build_context_batch(flux: torch.Tensor, flux_err: torch.Tensor, times: torch.Tensor,
                        metadata: torch.Tensor, k: int, C: int,
                        n_targets: int = 0, rng=None):
    """Build (N_valid, input_dim) context vectors for valid j in one sequence.

    Valid j range: [k + C, L - k - C) — positions with full context on both sides.
    Edge positions (closer than C points to a boundary on the gap-far side) are
    skipped to keep all inputs unpadded. This costs at most 2C positions per
    sequence (negligible for typical L >= 1000).

    Args:
        flux, flux_err, times: (L,) float tensors on the same device.
        metadata: (M,) tensor or None.
        k: prediction offset (int).
        C: context window size on each side.
        n_targets: if > 0, randomly subsample this many j positions per sequence.
                   0 means use all valid positions (slower but full signal).
        rng: numpy Generator used for the subsampling; required when n_targets > 0.

    Returns:
        x: (N, input_dim) tensor of context vectors, or None if no valid j.
        target_flux: (N,) tensor of true flux at the predicted positions.
        target_ferr: (N,) tensor of true flux_err at the predicted positions.
        target_idx: (N,) long tensor of j indices (useful for eval).
    """
    L = int(flux.shape[0])
    if L < 2 * (k + C) + 1:
        return None, None, None, None

    j_min = k + C
    j_max = L - k - C  # exclusive
    if j_max <= j_min:
        return None, None, None, None

    device = flux.device
    M = 0 if metadata is None else int(metadata.shape[0])

    n_full = j_max - j_min
    if n_targets and n_targets < n_full:
        if rng is None:
            raise ValueError("rng must be provided when n_targets > 0")
        picks = rng.choice(n_full, size=n_targets, replace=False)
        picks.sort()
        js = torch.from_numpy(picks.astype(np.int64)).to(device) + j_min
    else:
        js = torch.arange(j_min, j_max, device=device, dtype=torch.long)
    N = int(js.shape[0])

    arange_C = torch.arange(C, device=device, dtype=torch.long)
    fwd_idx = js.unsqueeze(1) - k - C + arange_C.unsqueeze(0)         # (N, C)
    bwd_idx = js.unsqueeze(1) + k + arange_C.unsqueeze(0)             # (N, C)

    fwd_flux = flux[fwd_idx]                                          # (N, C)
    bwd_flux = flux[bwd_idx]
    fwd_ferr = flux_err[fwd_idx]
    bwd_ferr = flux_err[bwd_idx]

    t_target = times[js].unsqueeze(1)                                 # (N, 1)
    fwd_dt = times[fwd_idx] - t_target
    bwd_dt = times[bwd_idx] - t_target

    # All positions valid here (we restricted j to keep full context). Carry the
    # mask channel anyway so the input shape is invariant if edge handling is
    # added later.
    valid_mask = torch.ones(N, 2 * C, device=device, dtype=flux.dtype)

    target_ferr = flux_err[js]                                        # (N,)
    log2_k = torch.full((N, 1), math.log2(max(k, 1)),
                        device=device, dtype=flux.dtype)

    parts = [
        fwd_flux, bwd_flux,
        fwd_ferr, bwd_ferr,
        fwd_dt, bwd_dt,
        valid_mask,
        target_ferr.unsqueeze(1),
        log2_k,
    ]
    if metadata is not None and M > 0:
        parts.append(metadata.unsqueeze(0).expand(N, M))

    x = torch.cat(parts, dim=1)                                       # (N, input_dim)
    target_flux = flux[js]
    return x, target_flux, target_ferr, js


def sample_k(K_max: int, L: int, rng) -> int:
    """Log-uniform k in [1, min(K_max, L//2 - C - 1)] (caller passes effective bound).

    Caller is responsible for ensuring the chosen k leaves at least one valid
    target position. Sampling here is purely the log-uniform draw.
    """
    if K_max < 1:
        return 1
    log_min = 0.0
    log_max = math.log10(max(K_max, 1))
    log_k = rng.uniform(log_min, log_max)
    k = int(round(10 ** log_k))
    return max(1, min(k, K_max))
