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

        num_time_enc_dims = 64
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
        # Small MLP head so the model can nonlinearly combine global summary
        # and per-step time encoding before predicting the scalar.
        head_hidden = max(32, hidden_size // 2)

        self.gauss_head = nn.Sequential(
            nn.Linear(self.output_size, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x, t):
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
            point_prediction = self.gauss_head(h_bi)
            seq_prediction.append(point_prediction)

        seq_prediction = torch.stack(seq_prediction, dim=1)     # (B, L, 1)
        return {'reconstructed': seq_prediction}
    
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

        # Defensive normalization for t sequence (B, L) or (B, L, 1)
        t_seq = t
        if t_seq.dim() == 3 and t_seq.size(-1) == 1:
            t_seq = t_seq.squeeze(-1)
        if t_seq.dim() != 2:
            raise ValueError("t must be shape (B, L) or (B, L, 1)")

        # Initial hidden states
        h_fwd = x.new_zeros(B, self.hidden_size)
        h_bwd = x.new_zeros(B, self.hidden_size)
        # ---- Store backward hidden state (mask-gated updates) ----
        h_bwd_tensor = x.new_zeros(B, L, self.hidden_size)
        for ti in reversed(range(L)):
            # store current hidden (before processing this timestep)
            h_bwd_tensor[:, ti, :] = h_bwd
            xi_bwd = x[:, ti, :]
            inp_bwd = self.backward_input_proj(xi_bwd)
            h_bwd_candidate = self.backward_cell.step(inp_bwd, h_bwd)

            mask_t = x[:, ti, 2].unsqueeze(-1)
            mask_bool = (mask_t > 0.5)
            mask_expand = mask_bool.expand(-1, self.hidden_size)
            h_bwd = torch.where(mask_expand, h_bwd_candidate, h_bwd)

        # User forward pass and predict (mask-gated updates)
        h_fwd_tensor = x.new_zeros(B, L, self.hidden_size)
        for ti in range(L):
            # store current hidden (before processing this timestep)
            h_fwd_tensor[:, ti, :] = h_fwd
            xi_fwd = x[:, ti, :]
            inp_fwd = self.forward_input_proj(xi_fwd)
            h_fwd_candidate = self.forward_cell.step(inp_fwd, h_fwd)

            mask_t = x[:, ti, 2].unsqueeze(-1)
            mask_bool = (mask_t > 0.5)
            mask_expand = mask_bool.expand(-1, self.hidden_size)
            h_fwd = torch.where(mask_expand, h_fwd_candidate, h_fwd)

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

        
        # For each predicted timestamp, select the last forward hidden state preceding
        # the timestamp and the first backward hidden state following the timestamp.
        # If timestamp is before the sequence, use h_bwd_final as the fallback.
        # If timestamp is after the sequence, use h_fwd_final as the fallback.
        # We'll do this per-batch sample since time arrays may differ per sample.

        # Compute final hidden fallbacks
        h_bwd_final = h_bwd_tensor[:, 0, :]
        h_fwd_final = h_fwd_tensor[:, -1, :]

        # t_seq already normalized above

        Tpred = t_pred_enc.size(1)
        for b in range(B):
            t_seq_b = t_seq[b]
            # searchsorted for this batch: returns indices in [0, L]
            indices = torch.searchsorted(t_seq_b, t_pred[b])
            for p in range(Tpred):
                idx_next = int(indices[p].item())
                idx_prev = idx_next - 1

                if idx_prev >= 0:
                    h_fwd_sel = h_fwd_tensor[b, idx_prev, :]
                else:
                    # timestamp before sequence -> use final backward hidden
                    h_fwd_sel = h_bwd_final[b]

                if idx_next < L:
                    h_bwd_sel = h_bwd_tensor[b, idx_next, :]
                else:
                    # timestamp after sequence -> use final forward hidden
                    h_bwd_sel = h_fwd_final[b]

                time_enc_p = t_pred_enc[b, p, :]
                h_bi = torch.cat([h_fwd_sel, h_bwd_sel, time_enc_p], dim=0).unsqueeze(0)
                pred_bi = self.gauss_head(h_bi)
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
