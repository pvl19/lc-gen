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

        num_time_enc_dims = 64
        # Non-linear time encoder: scalar time -> richer time features.
        # Small MLP lets the head exploit time information more flexibly.
        self.time_enc = nn.Sequential(
            nn.Linear(1, num_time_enc_dims),
            nn.ReLU(),
            nn.Linear(num_time_enc_dims, num_time_enc_dims),
        )

        # We'll accept scalar flux + flux_err + mask = time encoding inputs per timestep (2D)
        self.forward_input_proj = nn.Linear(3 + num_time_enc_dims, hidden_size)
        self.backward_input_proj = nn.Linear(3 + num_time_enc_dims, hidden_size)

        # forward and backward minGRU cells
        self.forward_cell = minGRUCell(hidden_size, hidden_size)
        self.backward_cell = minGRUCell(hidden_size, hidden_size)

        self.output_size = hidden_size * 2 + num_time_enc_dims
        # Small MLP head so the model can nonlinearly combine global summary
        # and per-step time encoding before predicting the scalar.
        head_hidden = max(self.output_size // 2, 128)
        self.gauss_head = nn.Sequential(
            nn.Linear(self.output_size, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # Learnable mask token for masked timesteps: replaces (flux, flux_err)
        # when a timestep is masked. We keep the mask channel itself so the
        # model can still detect masked positions if needed.
        # Shape: (2,) -> [flux_token, flux_err_token]
        means = torch.tensor([0.0, 0.01])
        stds  = torch.tensor([0.1, 0.001])
        self.mask_token = nn.Parameter(torch.normal(means, stds))

    def forward(self, x, t):
        if x.dim() != 3 or x.size(-1) != 3:
            raise ValueError(f"Expected (B, L, 3), got {tuple(x.shape)}")

        B, L, _ = x.shape

        # Replace masked timesteps' (flux, flux_err) with a learned mask token.
        # Dataset convention: mask == 1 -> unmasked, mask == 0 -> masked.
        mask_chan = x[..., 2:3]                    # (B, L, 1)
        masked_pos = (mask_chan == 0)              # boolean (B, L, 1)
        if masked_pos.any():
            token = self.mask_token.view(1, 1, 2).to(x.device)   # (1,1,2)
            # Replace only the first two channels (flux, flux_err)
            x_first2 = x[..., :2]
            x_first2 = torch.where(masked_pos.expand_as(x_first2), token, x_first2)
            x = torch.cat([x_first2, mask_chan], dim=-1)

        t_enc = self.time_enc(t)  # (B, L, 16)
        x = torch.cat([x, t_enc], dim=-1)  # (B, L, 3 + 16)

        # Preallocate backward hidden states
        h_bwd_tensor = x.new_zeros(B, L, self.hidden_size)
        h_fwd_tensor = x.new_zeros(B, L, self.hidden_size)

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
            # Store forward hidden state
            h_fwd_tensor[:, t, :] = h_fwd           # store aligned with original sequence

            # Forward RNN
            xi_fwd = x[:, t, :]                     # (B, input_dim)
            inp_fwd = self.forward_input_proj(xi_fwd)  # (B, H)
            h_fwd = self.forward_cell.step(inp_fwd, h_fwd)            

        h_bwd_final = h_bwd_tensor[:, 0, :]
        h_fwd_final = h_fwd_tensor[:, -1, :]

        for t in range(L):

            # Get the mask for this timestep: (B,)
            mask_t = x[:, t, 2].unsqueeze(-1)     # → (B, 1)

            # Compute candidate hidden states
            h_fwd_unmasked = h_fwd_tensor[:, t, :]   # (B, H)
            h_fwd_masked = h_fwd_final               # (B, H)
            h_bwd_unmasked = h_bwd_tensor[:, t, :]   # (B, H)
            h_bwd_masked = h_bwd_final               # (B, H)

            # Per-sample selection (vectorized)
            h_fwd_apply = torch.where(mask_t > 0, h_fwd_unmasked, h_fwd_masked)
            h_bwd_apply = torch.where(mask_t > 0, h_bwd_unmasked, h_bwd_masked)

            time_enc_t = t_enc[:, t, :]

            # Concatenate and predict
            h_bi = torch.cat([h_fwd_apply, h_bwd_apply, time_enc_t], dim=1)
            pred_bi = self.gauss_head(h_bi)
            preds_bi.append(pred_bi)

        preds_bi = torch.stack(preds_bi, dim=1)     # (B, L, 1)
        return {'reconstructed': preds_bi}
    
    def predict(self, x, t, t_pred):
        if x.dim() != 3 or x.size(-1) != 3:
            raise ValueError(f"Expected (B, L, 3), got {tuple(x.shape)}")
        
        B, L, _ = x.shape
        # Replace masked timesteps' (flux, flux_err) with the learned mask token
        # (keep mask channel unchanged). Dataset convention: mask == 1 -> unmasked
        mask_chan = x[..., 2:3]
        masked_pos = (mask_chan == 0)
        if masked_pos.any():
            token = self.mask_token.view(1, 1, 2).to(x.device)
            x_first2 = x[..., :2]
            x_first2 = torch.where(masked_pos.expand_as(x_first2), token, x_first2)
            x = torch.cat([x_first2, mask_chan], dim=-1)

        t_enc = self.time_enc(t)  # (B, L, 16)
        x = torch.cat([x, t_enc], dim=-1)  # (B, L, 3 + 16)

        # Initial hidden states
        h_fwd = x.new_zeros(B, self.hidden_size)
        h_bwd = x.new_zeros(B, self.hidden_size)

        # ---- Store backward hidden state ----
        for ti in reversed(range(L)):
            # Backward RNN
            xi_bwd = x[:, ti, :]
            inp_bwd = self.backward_input_proj(xi_bwd)
            h_bwd = self.backward_cell.step(inp_bwd, h_bwd)

        # User forward pass and predict
        for ti in range(L):
            # Forward RNN
            xi_fwd = x[:, ti, :]                     # (B, input_dim)
            inp_fwd = self.forward_input_proj(xi_fwd)  # (B, H)
            h_fwd = self.forward_cell.step(inp_fwd, h_fwd)            

        preds_bi = []

        # ---- Encode predicted timestamps ----
        # t_pred may be (T_pred,) or (B, T_pred)
        t_pred = torch.tensor(t_pred, device=x.device, dtype=x.dtype)

        if t_pred.dim() == 1:
            # User gave a single timeline → broadcast
            t_pred = t_pred.unsqueeze(0).expand(B, -1)
        elif t_pred.dim() == 2:
            # Already (B, T_pred) → do nothing
            pass
        else:
            raise ValueError("t_pred must be 1D or 2D (B, T_pred)")
        t_pred_enc = self.time_enc(t_pred.unsqueeze(-1))                   # (B,T_pred,16)

        
        #Concatenate hidden states and predict
        for t in range(t_pred_enc.size(1)):
            h_bi = torch.cat([h_fwd, h_bwd, t_pred_enc[:, t, :]], dim=1)  # (B, 2H + time_enc_dims)
            pred_bi = self.gauss_head(h_bi)  # (B,1)
            preds_bi.append(pred_bi)

        preds_bi = torch.stack(preds_bi, dim=1)     # (B, T_pred, 1)
        return {'predicted': preds_bi}

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
