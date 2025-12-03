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

class BiDirectionalMinGRU(nn.Module):
    """Bidirectional minGRU model."""
    def __init__(self, hidden_size: int = 64, use_flow: bool = False):
        super().__init__()
        self.hidden_size = hidden_size

        # We'll accept scalar flux + flux_err inputs per timestep (2D)
        self.forward_input_proj = nn.Linear(3, hidden_size)
        self.backward_input_proj = nn.Linear(3, hidden_size)

        # forward and backward minGRU cells
        self.forward_cell = minGRUCell(hidden_size, hidden_size)
        self.backward_cell = minGRUCell(hidden_size, hidden_size)

        self.output_size = hidden_size * 2
        self.gauss_head = nn.Linear(self.output_size, 1)

    def forward(self, x):
        if x.dim() != 3 or x.size(-1) != 3:
            raise ValueError(f"Expected (B, L, 3), got {tuple(x.shape)}")
        
        B, L, _ = x.shape

        # Preallocate hidden states
        h_fwd_tensor = x.new_zeros(B, L, self.hidden_size)
        h_bwd_tensor = x.new_zeros(B, L, self.hidden_size)

        # Initial hidden states
        h_fwd = x.new_zeros(B, self.hidden_size)
        h_bwd = x.new_zeros(B, self.hidden_size)

        preds_bi = []

        # ---- Store backward hidden states ----
        for t in reversed(range(L)):
            # Store hidden state
            h_bwd_tensor[:, t, :] = h_bwd           # store aligned with original sequence

            # Backward RNN
            xi_bwd = x[:, t, :]
            inp_bwd = self.backward_input_proj(xi_bwd)
            h_bwd = self.backward_cell.step(inp_bwd, h_bwd)

        # User forward pass and predict
        for t in range(L):

            #Concatenate hidden states and predict
            h_bwd = h_bwd_tensor[:, t, :]
            h_bi = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2H)
            pred_bi = self.gauss_head(h_bi)  # (B,1)
            preds_bi.append(pred_bi)

            # Forward RNN
            xi_fwd = x[:, t, :]                     # (B, input_dim)
            inp_fwd = self.forward_input_proj(xi_fwd)  # (B, H)
            h_fwd = self.forward_cell.step(inp_fwd, h_fwd)            

        preds_bi = torch.stack(preds_bi, dim=1)     # (B, L, 1)
        return {'reconstructed': preds_bi}

class SimpleMinGRU(nn.Module):
    """Single-cell RNN model with Gaussian and optional Flow heads."""
    def __init__(self, hidden_size: int = 64, direction: str = 'forward', use_flow: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.direction = direction

        # We'll accept scalar flux + flux_err inputs per timestep (2D)
        self.input_proj = nn.Linear(3, hidden_size)

        # single minGRU cell
        self.cell = minGRUCell(hidden_size, hidden_size)

        # Output projections
        # Gaussian head: outputs [mean]
        self.gauss_head = nn.Linear(hidden_size, 1)

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

    def forward(self, x):
        if x.dim() != 3 or x.size(-1) != 3:
            raise ValueError(f"Expected (B, L, 3), got {tuple(x.shape)}")

        B, L, _ = x.shape
        h = x.new_zeros(B, self.hidden_size)

        if self.direction == 'backward':
            x = torch.flip(x, dims=[1])  # reverse sequence for backward pass

        preds = []

        for t in range(L):
            
            # Predict next value using current state
            pred = self.gauss_head(h)    # (B,1)
            preds.append(pred)
            
            # Feed current true timestep into RNN
            xi = x[:, t, :]                  # (B,3)
            inp = self.input_proj(xi)         # (B,H)
            h = self.cell.step(inp, h)        # update state

        # preds = list of L items [(B,1), (B,1), ...]
        preds = torch.stack(preds, dim=1)     # (B, L, 1)

        if self.direction == 'backward':
            preds = torch.flip(preds, dims=[1])  # reverse predictions back

        return {'reconstructed': preds}

def example_usage():
    print("Example usage of SimpleMinGRU model.")
    m = SimpleMinGRU(hidden_size=32, use_flow=(zuko is not None))
    x = torch.randn(2, 128)
    out = m(x)
    print('recon', out['reconstructed'].shape, 'flow_ctx', out['flow_context'].shape)


if __name__ == '__main__':
    example_usage()
