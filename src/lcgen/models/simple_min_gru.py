"""Simple single-cell minGRU model with Gaussian and optional Flow heads.

This minimal model keeps a single-time-step minGRU cell used sequentially
to process a 1D time series and provides two output heads:
 - Gaussian head: per-timestep mean and raw log-sigma (pre-softplus)
 - Flow head (optional): conditional normalizing flow (zuko) for p(y|ctx)

API:
  model(x) -> dict with keys:
    'reconstructed': Tensor (B, L, 2) -> [mean, raw_logsigma]
    'flow_context': Tensor (B, L, C) if flow enabled else None

This file is intentionally small and self-contained.
"""
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import zuko
except Exception:
    zuko = None


class minGRUCell(nn.Module):
    """Minimal GRU cell (single-step) compatible with the project's minGRU."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_z = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(input_size, hidden_size)

    def step(self, x_t, h_prev=None):
        if h_prev is None:
            h_prev = x_t.new_zeros(x_t.shape[0], self.W_h.out_features)
        z = torch.sigmoid(self.W_z(x_t))
        h_tilde = torch.tanh(self.W_h(x_t))
        h = (1.0 - z) * h_prev + z * h_tilde
        return h


class SimpleMinGRU(nn.Module):
    """Single-cell RNN model with Gaussian and optional Flow heads."""
    def __init__(self, hidden_size: int = 64, use_flow: bool = True):
        super().__init__()
        self.hidden_size = hidden_size

        # We'll accept scalar flux inputs per timestep (1D)
        self.input_proj = nn.Linear(1, hidden_size)

        # single minGRU cell
        self.cell = minGRUCell(hidden_size, hidden_size)

        # Output projections
        # Gaussian head: outputs [mean, raw_logsigma]
        self.gauss_head = nn.Linear(hidden_size, 2)

        # Flow context projection
        self.flow_ctx_proj = nn.Linear(hidden_size, hidden_size)

        # Optional flow posterior
        self.flow = None
        if use_flow:
            if zuko is None:
                # If zuko not installed, keep flow None but don't raise here.
                self.flow = None
            else:
                # small NSF for 1D outputs conditioned on context
                self.flow = zuko.flows.NSF(1, hidden_size, transforms=2, hidden_features=[64, 64])

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: (B, L) or (B, L, 1)
        Returns:
            dict with 'reconstructed' (B, L, 2) and 'flow_context' (B, L, C) or None
        """
        if x.dim() == 2:
            x_in = x.unsqueeze(-1)
        else:
            x_in = x

        B, L, _ = x_in.shape
        h = x_in.new_zeros(B, self.hidden_size)

        means = []
        logs = []
        flow_ctxs = []

        for t in range(L):
            xi = x_in[:, t:t+1, :].squeeze(1)  # (B, 1)
            inp = self.input_proj(xi)  # (B, H)
            h = self.cell.step(inp, h)

            out = self.gauss_head(h)
            mean_t = out[:, 0]
            logsig_raw_t = out[:, 1]
            means.append(mean_t.unsqueeze(1))
            logs.append(logsig_raw_t.unsqueeze(1))

            ctx = self.flow_ctx_proj(h)
            flow_ctxs.append(ctx.unsqueeze(1))

        means = torch.cat(means, dim=1)
        logs = torch.cat(logs, dim=1)
        recon = torch.stack([means, logs], dim=-1)

        flow_ctx = torch.cat(flow_ctxs, dim=1)

        return {'reconstructed': recon, 'flow_context': flow_ctx}


def example_usage():
    m = SimpleMinGRU(hidden_size=32, use_flow=(zuko is not None))
    x = torch.randn(2, 128)
    out = m(x)
    print('recon', out['reconstructed'].shape, 'flow_ctx', out['flow_context'].shape)


if __name__ == '__main__':
    example_usage()
