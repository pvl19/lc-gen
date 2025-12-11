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
    def __init__(self, hidden_size: int = 64, direction: str = "bi", use_flow: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.direction = direction

        num_time_enc_dims = 8
        self.num_time_enc_dims = num_time_enc_dims
        # Non-linear time encoder: scalar time -> richer time features.
        # Small MLP lets the head exploit time information more flexibly.
        self.time_enc = nn.Sequential(
            nn.Linear(1, num_time_enc_dims),
            nn.ReLU(),
            nn.Linear(num_time_enc_dims, num_time_enc_dims),
        )

        # We'll accept scalar flux + flux_err + mask = time encoding inputs per timestep (2D)
        self.forward_input_proj = nn.Linear(2 + num_time_enc_dims, hidden_size)
        self.backward_input_proj = nn.Linear(2 + num_time_enc_dims, hidden_size)

        # forward and backward minGRU cells
        self.forward_cell = minGRUCell(hidden_size, hidden_size)
        self.backward_cell = minGRUCell(hidden_size, hidden_size)

        if self.direction == 'bi':
            self.output_size = hidden_size * 2 + num_time_enc_dims
        else:
            self.output_size = hidden_size + num_time_enc_dims
        # single LayerNorm for head inputs (defined after output_size)
        self.head_norm = nn.LayerNorm(self.output_size)
        # initialize time_scale to 2.0 to give the time-encoding a larger initial
        # contribution and make it easier for the head to rely on absolute time.
        self.time_scale = nn.Parameter(torch.tensor(5.0))
        # Small MLP head so the model can nonlinearly combine global summary
        # and per-step time encoding before predicting the scalar.
        head_hidden = max(32, hidden_size // 2)

        self.gauss_head = nn.Sequential(
            nn.Linear(self.output_size, head_hidden),
            # use LeakyReLU so the head can produce negative activations easily
            # (prevents the final linear from only seeing non-negative inputs)
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x, t, return_states: bool = False):
        if x.dim() != 3 or x.size(-1) != 2:
            raise ValueError(f"Expected (B, L, 2), got {tuple(x.shape)}")

        B, L, _ = x.shape

        # Defensive normalization: expect t to be (B, L) or (B, L, 1)
        t_seq = t
        if t_seq.dim() == 3 and t_seq.size(-1) == 1:
            t_seq = t_seq.squeeze(-1)
        if t_seq.dim() != 2:
            raise ValueError("t must be shape (B, L) or (B, L, 1)")

        t_enc = self.time_enc(t)  # (B, L, 16)
        x = torch.cat([x, t_enc], dim=-1)  # (B, L, 2 + 16)

        seq_prediction = []


        # ---- Store backward hidden states (mask-gated updates) ----

        if self.direction in ['bi', 'backward']:
            h_bwd_tensor = x.new_zeros(B, L, self.hidden_size)
            h_bwd = x.new_zeros(B, self.hidden_size)
            for ti in reversed(range(L)):
                # Store hidden state (before processing this timestep)
                h_bwd_tensor[:, ti, :] = h_bwd

                # Backward RNN step for this timestep
                xi_bwd = x[:, ti, :]
                inp_bwd = self.backward_input_proj(xi_bwd)
                h_bwd = self.backward_cell.step(inp_bwd, h_bwd)

        if self.direction in ['bi', 'forward']:
            h_fwd_tensor = x.new_zeros(B, L, self.hidden_size)
            h_fwd = x.new_zeros(B, self.hidden_size)

            for ti in range(L):
                h_fwd_tensor[:, ti, :] = h_fwd

                # Forward RNN step
                xi_fwd = x[:, ti, :]
                inp_fwd = self.forward_input_proj(xi_fwd)
                h_fwd = self.forward_cell.step(inp_fwd, h_fwd)

        for ti in range(L):
            # Use closest hidden states available from unmasked data at this index
            time_enc_t = t_enc[:, ti, :]

            # Concatenate and predict
            if self.direction == 'forward':
                h_fwd_apply = h_fwd_tensor[:, ti, :]
                h_bi = torch.cat([h_fwd_apply, time_enc_t], dim=1)
            elif self.direction == 'backward':
                h_bwd_apply = h_bwd_tensor[:, ti, :]
                h_bi = torch.cat([h_bwd_apply, time_enc_t], dim=1)
            else:
                h_fwd_apply = h_fwd_tensor[:, ti, :]
                h_bwd_apply = h_bwd_tensor[:, ti, :]
                h_bi = torch.cat([h_fwd_apply, h_bwd_apply, time_enc_t], dim=1)
            # normalize fused vector so time and hidden parts have comparable scale
            h_bi = self.head_norm(h_bi)
            # Apply a learnable scalar to the time-encoding portion AFTER LayerNorm
            # so the scalar is not normalized away by head_norm.
            nt = self.num_time_enc_dims
            if nt > 0:
                # multiply only the final nt dimensions (the time encoding slice)
                h_hidden = h_bi[:, :-nt]
                h_time = h_bi[:, -nt:] * self.time_scale
                h_bi = torch.cat([h_hidden, h_time], dim=1)
            point_prediction = self.gauss_head(h_bi)
            seq_prediction.append(point_prediction)

        seq_prediction = torch.stack(seq_prediction, dim=1)     # (B, L, 1)
        out = {'reconstructed': seq_prediction}
        if return_states:
            # Provide forward hidden states and time encodings so training code can
            # compute multi-step losses based on the hidden states BEFORE each timestep.
            if self.direction in ['bi', 'forward']:
                out['h_fwd_tensor'] = h_fwd_tensor
            if self.direction in ['bi', 'backward']:
                out['h_bwd_tensor'] = h_bwd_tensor
            out['t_enc'] = t_enc
        return out

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
