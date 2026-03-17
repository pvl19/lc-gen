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

from lcgen.models.MetadataAgePredictor import MetadataEncoder

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

    def g(x):
        return torch.where(x >= 0, x+0.5, torch.sigmoid(x))
    
    def log_g(x):
        return torch.where(x >= 0, (F.relu(x)+0.5).log(), 5 -F.softplus(-x))

    def parallel_scan_log(log_coeffs, log_values):
        # log_coeffs: (batch_size, seq_len, input_size)
        # log_values: (batch_size, seq_len + 1, input_size)
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h)[:, 1:]

    def step(self, x_t, h_prev=None):
        if h_prev is None:
            h_prev = x_t.new_zeros(x_t.shape[0], self.W_h.out_features)
        z = torch.sigmoid(self.W_z(x_t))
        h_tilde = torch.tanh(self.W_h(x_t))
        h = (1.0 - z) * h_prev + z * h_tilde
        return h
    
    def parallel_scan(self, a, b):
        """
        a: (B, T, H)        multiplicative terms (should be in (0,1] for GRU)
        b: (B, T+1, H)      additive terms, with b[:, 0] = h0
        returns:
        h: (B, T, H)
        
        Solves the linear recurrence h_t = a_t * h_{t-1} + b_t for t=1..T.
        
        This implementation uses associative scan which is O(T) depth but
        parallelizable. Falls back to sequential for very long sequences.
        """
        B, T, H = a.shape
        
        # For the recurrence h_t = a_t * h_{t-1} + b_t, we can write this as
        # a matrix operation and use associative scan. Each step is:
        # [h_t, 1] = [[a_t, b_t], [0, 1]] @ [h_{t-1}, 1]
        # 
        # The simplest numerically stable approach for moderate T is sequential.
        # For very large T, a proper associative scan should be implemented.
        
        h0 = b[:, 0, :]  # (B, H)
        b_seq = b[:, 1:, :]  # (B, T, H) - the b_1..b_T terms
        
        h_list = []
        h_prev = h0
        for t in range(T):
            h_t = a[:, t, :] * h_prev + b_seq[:, t, :]
            h_list.append(h_t)
            h_prev = h_t
        
        return torch.stack(h_list, dim=1)  # (B, T, H)
    
    def step_parallel(self, x, h_0):
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)

        z = torch.sigmoid(self.W_z(x))           # (B, T, H)
        h_tilde = torch.tanh(self.W_h(x))        # (B, T, H) - must match step() which uses tanh

        a = 1.0 - z
        b = torch.cat([h_0, z * h_tilde], dim=1)

        h = self.parallel_scan(a, b)
        return h
    
    def step_parallel_log(self, x, h_0):
        """Parallelized step for batch processing."""
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)
        k = self.linear_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h_0 = self.log_g(h_0)
        log_tilde_h = self.log_g(self.linear_h(x))
        h = self.parallel_scan_log(log_coeffs,torch.cat([log_h_0, log_z + log_tilde_h], dim=1))
        return h
        

class BiDirectionalMinGRU(nn.Module):
    """Bidirectional minGRU model."""
    def __init__(
        self,
        hidden_size: int = 64,
        direction: str = "bi",
        mode: str = "sequential",
        use_flow: bool = False,
        num_meta_features: int = 13,
        use_conv_channels: bool = False,
        conv_config: dict = None,
        meta_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_flow = use_flow
        self.direction = direction
        self.mode = mode
        self.use_conv_channels = use_conv_channels

        num_time_enc_dims = 8
        self.num_time_enc_dims = num_time_enc_dims
        # Non-linear time encoder: scalar time -> richer time features.
        # Small MLP lets the head exploit time information more flexibly.
        self.time_enc = nn.Sequential(
            nn.Linear(1, num_time_enc_dims),
            nn.ReLU(),
            nn.Linear(num_time_enc_dims, num_time_enc_dims),
        )

        # Stellar metadata encoder: shared MetadataEncoder from MetadataAgePredictor.
        # Projects static stellar properties to a learned embedding broadcast to every
        # RNN timestep as additional context. 3 hidden layers × 128 dims, output 32 dims.
        self.num_meta_features = num_meta_features
        if num_meta_features > 0:
            meta_emb_dim = 32
            self.meta_encoder = MetadataEncoder(
                input_dim=num_meta_features,
                latent_dim=meta_emb_dim,
                hidden_dims=[128, 128, 128],
                dropout=meta_dropout,
                use_mask=False,
            )
        else:
            meta_emb_dim = 0
            self.meta_encoder = None
        self.meta_emb_dim = meta_emb_dim

        # Convolutional encoders for power spectra and ACF
        # These encode global frequency-domain features of the light curve
        if use_conv_channels:
            # Default conv config if not provided
            if conv_config is None:
                conv_config = {
                    'encoder_type': 'lightweight',  # 'lightweight' or 'unet'
                    'hidden_channels': 16,
                    'num_layers': 3,
                    'activation': 'gelu'
                }

            encoder_type = conv_config.get('encoder_type', 'lightweight')

            if encoder_type == 'lightweight':
                # Use fast lightweight encoder (recommended)
                from .lightweight_conv import LightweightConv1DEncoder

                # Encoder for power spectrum + f-statistic (2 channels)
                self.ps_encoder = LightweightConv1DEncoder(
                    in_channels=2,
                    hidden_channels=conv_config.get('hidden_channels', 16),
                    num_layers=conv_config.get('num_layers', 3),
                    activation=conv_config.get('activation', 'gelu')
                )

                # Encoder for autocorrelation function (1 channel)
                self.acf_encoder = LightweightConv1DEncoder(
                    in_channels=1,
                    hidden_channels=conv_config.get('hidden_channels', 16),
                    num_layers=conv_config.get('num_layers', 3),
                    activation=conv_config.get('activation', 'gelu')
                )

                conv_bottleneck_channels = self.ps_encoder.output_channels

            else:
                # Use UNet encoder (slower but more expressive)
                from .conv_models import PowerSpectrumUNetEncoder

                # Create config object
                class ConvArgs:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)

                # Encoder for power spectrum + f-statistic (2 channels)
                conv_args_ps = ConvArgs(
                    input_length=conv_config.get('input_length', 16000),
                    encoder_dims=conv_config.get('encoder_dims', [4, 8, 16, 32]),
                    num_layers=conv_config.get('num_layers', 4),
                    activation=conv_config.get('activation', 'gelu'),
                    in_channels=2
                )
                self.ps_encoder = PowerSpectrumUNetEncoder(conv_args_ps)

                # Encoder for autocorrelation function (1 channel)
                conv_args_acf = ConvArgs(
                    input_length=conv_config.get('input_length', 16000),
                    encoder_dims=conv_config.get('encoder_dims', [4, 8, 16, 32]),
                    num_layers=conv_config.get('num_layers', 4),
                    activation=conv_config.get('activation', 'gelu'),
                    in_channels=1
                )
                self.acf_encoder = PowerSpectrumUNetEncoder(conv_args_acf)

                conv_bottleneck_channels = conv_config['encoder_dims'][-1]

            # Use global average pooling on the bottleneck to get one value per channel
            # This reduces [batch, channels, spatial] -> [batch, channels] via mean pooling
            # Much more efficient than flattening all spatial features!

            # Project pooled conv outputs to embedding dimension (per channel)
            conv_emb_dim = 32  # embedding dimension for each conv channel
            self.ps_conv_proj = nn.Sequential(
                nn.Linear(conv_bottleneck_channels, conv_emb_dim),
                nn.ReLU(),
                nn.Linear(conv_emb_dim, conv_emb_dim)
            )
            self.acf_conv_proj = nn.Sequential(
                nn.Linear(conv_bottleneck_channels, conv_emb_dim),
                nn.ReLU(),
                nn.Linear(conv_emb_dim, conv_emb_dim)
            )
            self.conv_emb_dim = conv_emb_dim * 2  # Total conv embedding size (PS + ACF)
        else:
            self.ps_encoder = None
            self.acf_encoder = None
            self.ps_conv_proj = None
            self.acf_conv_proj = None
            self.conv_emb_dim = 0

        # We'll accept scalar flux + flux_err + time encoding + metadata embedding + conv embeddings per timestep
        rnn_input_dim = 2 + num_time_enc_dims + meta_emb_dim + self.conv_emb_dim
        self.forward_input_proj = nn.Linear(rnn_input_dim, hidden_size)
        self.backward_input_proj = nn.Linear(rnn_input_dim, hidden_size)

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
        # Optional conditional normalizing flow head (zuko). If zuko is
        # available, construct a small Neural Spline Flow (NSF) that models
        # p(y | context) where context = [hidden(s), time_enc_slice, meas_err].
        self.flow = None
        if zuko is not None and use_flow:
            # context dim = size of head input after head_norm/time-scaling + 1 for meas_err
            flow_context_dim = self.output_size + 1
            # small NSF: 2 transforms, small hidden networks
            self.flow = zuko.flows.NSF(1, flow_context_dim, transforms=2, hidden_features=[64, 64])

    def forward(self, x, t, mask=None, metadata=None, conv_data=None, return_states: bool = False, flow_mode: str = 'mean'):
        """Forward pass through the model.

        Args:
            x: (B, L, 2) input tensor [flux, flux_err]
            t: (B, L) or (B, L, 1) timestamps
            mask: (B, L) optional sequence mask (1=observed, 0=masked/padded).
                  Masked positions have their flux/flux_err zeroed in the RNN input.
            metadata: (B, num_meta_features) optional stellar metadata tensor.
                      If None, the metadata path is skipped.
            conv_data: dict with optional convolutional inputs:
                      - 'power': (B, freq_length) power spectrum
                      - 'f_stat': (B, freq_length) f-statistic
                      - 'acf': (B, freq_length) autocorrelation function
                      If None or use_conv_channels=False, zeros are used.
            return_states: if True, return hidden states and time encodings
            flow_mode: how to produce point predictions when flow head is present.
                - 'mean': Monte Carlo average of N samples (default, good for reconstruction)
                - 'mode': gradient-based optimization to find the flow mode (MAP estimate)
                - 'sample': single random sample from the flow (stochastic)
                Ignored when flow head is not present (uses Gaussian head).

        Returns:
            dict with 'reconstructed' (B, L, 1), optionally hidden states and t_enc
        """
        if x.dim() != 3 or x.size(-1) != 2:
            raise ValueError(f"Expected (B, L, 2), got {tuple(x.shape)}")

        B, L, _ = x.shape

        # Apply mask to input: zero out flux and flux_err at masked positions
        # but keep time encoding so the RNN knows the timestamp
        if mask is not None:
            # mask shape: (B, L) -> expand to (B, L, 1) for broadcasting
            mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
            # Zero out flux (index 0) and flux_err (index 1) at masked positions
            x = x * mask_expanded

        # Defensive normalization: expect t to be (B, L) or (B, L, 1)
        t_seq = t
        if t_seq.dim() == 3 and t_seq.size(-1) == 1:
            # make (B, L)
            t_seq = t_seq.squeeze(-1)
        if t_seq.dim() != 2:
            raise ValueError("t must be shape (B, L) or (B, L, 1)")

        # Shift each sequence so times start at zero: t <- t - t0 where t0 = t[:,0]
        # Keep units unchanged; this ensures the time-encoder sees time relative to
        # the start of the sequence.
        t0 = t_seq[:, 0].unsqueeze(1)  # (B, 1)
        t_shifted = t_seq - t0         # (B, L)
        # time_enc expects a last-dim scalar, so restore (...,1)
        t_enc = self.time_enc(t_shifted.unsqueeze(-1))  # (B, L, Te)

        # Encode stellar metadata (computed once, broadcast to every RNN timestep).
        if self.meta_encoder is not None and metadata is not None:
            meta_emb = self.meta_encoder(metadata)  # (B, meta_emb_dim)
            meta_emb_seq = meta_emb.unsqueeze(1).expand(-1, L, -1)  # (B, L, meta_emb_dim)
        else:
            meta_emb = None
            meta_emb_seq = None

        # Encode convolutional channels (power spectrum + f-stat, and ACF)
        # These provide global frequency-domain context for the light curve
        if self.use_conv_channels and conv_data is not None:
            # Encode power spectrum + f-statistic
            if 'power' in conv_data and 'f_stat' in conv_data:
                power = conv_data['power']  # (B, freq_length)
                f_stat = conv_data['f_stat']  # (B, freq_length)

                # Stack as 2-channel input: (B, 2, freq_length)
                ps_input = torch.stack([power, f_stat], dim=1)
                ps_result = self.ps_encoder(ps_input)  # Can be tensor or tuple
                ps_encoded = ps_result[0] if isinstance(ps_result, tuple) else ps_result
                # Apply global average pooling: (B, channels, spatial) -> (B, channels)
                ps_pooled = ps_encoded.mean(dim=-1)  # Much more efficient than flattening!
                ps_emb = self.ps_conv_proj(ps_pooled)  # (B, conv_emb_dim/2)
            else:
                ps_emb = torch.zeros(B, self.conv_emb_dim // 2, device=x.device)

            # Encode autocorrelation function
            if 'acf' in conv_data:
                acf = conv_data['acf']  # (B, freq_length)
                acf_result = self.acf_encoder(acf)  # Can be tensor or tuple
                acf_encoded = acf_result[0] if isinstance(acf_result, tuple) else acf_result
                # Apply global average pooling: (B, channels, spatial) -> (B, channels)
                acf_pooled = acf_encoded.mean(dim=-1)
                acf_emb = self.acf_conv_proj(acf_pooled)  # (B, conv_emb_dim/2)
            else:
                acf_emb = torch.zeros(B, self.conv_emb_dim // 2, device=x.device)

            # Combine both conv embeddings
            conv_emb = torch.cat([ps_emb, acf_emb], dim=-1)  # (B, conv_emb_dim)
            # Expand to sequence length for RNN input
            conv_emb_seq = conv_emb.unsqueeze(1).expand(-1, L, -1)  # (B, L, conv_emb_dim)
        else:
            conv_emb = None
            conv_emb_seq = None

        # Concatenate all embeddings to RNN input
        x_parts = [x, t_enc]
        if meta_emb_seq is not None:
            x_parts.append(meta_emb_seq)
        if conv_emb_seq is not None:
            x_parts.append(conv_emb_seq)
        x = torch.cat(x_parts, dim=-1)  # (B, L, 2 + Te + meta_emb_dim + conv_emb_dim)

        seq_prediction = []


        # ---- Store backward hidden states (mask-gated updates) ----

        if self.direction in ['bi', 'backward']:
            h_bwd_tensor = x.new_zeros(B, L, self.hidden_size)
            h_bwd = x.new_zeros(B, self.hidden_size)

            if self.mode == 'parallel':
                inp_bwd = x.flip(dims=[1])  # reverse sequence for backward pass
                x_bwd_proj = self.backward_input_proj(inp_bwd)
                h_bwd_all = self.backward_cell.step_parallel(x_bwd_proj, h_bwd.unsqueeze(1))
                h_bwd_tensor = h_bwd_all.flip(dims=[1])
                h0_b = h_bwd_tensor.new_zeros(B, 1, self.hidden_size)
                h_bwd_tensor = torch.cat([h_bwd_tensor[:, 1:, :], h0_b], dim=1)

            elif self.mode == 'sequential':
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

            if self.mode == 'parallel':
                inp_fwd = x
                x_fwd_proj = self.forward_input_proj(inp_fwd)
                h_fwd_all = self.forward_cell.step_parallel(x_fwd_proj, h_fwd.unsqueeze(1))
                h_fwd_tensor = h_fwd_all
                h0_f = h_fwd_tensor.new_zeros(B, 1, self.hidden_size)
                h_fwd_tensor = torch.cat([h0_f, h_fwd_tensor[:, :-1, :]], dim=1)

            elif self.mode == 'sequential':
                for ti in range(L):
                    h_fwd_tensor[:, ti, :] = h_fwd

                    # Forward RNN step
                    xi_fwd = x[:, ti, :]
                    inp_fwd = self.forward_input_proj(xi_fwd)
                    h_fwd = self.forward_cell.step(inp_fwd, h_fwd)

        for ti in range(L):
            # Use closest hidden states available from unmasked data at this index
            time_enc_t = t_enc[:, ti, :]

            # Concatenate hidden states + time encoding for head input.
            # Note: metadata and conv embeddings only affect the RNN hidden states,
            # not the prediction head input directly.
            if self.direction == 'forward':
                h_bi = torch.cat([h_fwd_tensor[:, ti, :], time_enc_t], dim=1)
            elif self.direction == 'backward':
                h_bi = torch.cat([h_bwd_tensor[:, ti, :], time_enc_t], dim=1)
            else:
                h_bi = torch.cat([h_fwd_tensor[:, ti, :], h_bwd_tensor[:, ti, :], time_enc_t], dim=1)
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
            # measurement error at this timestep is present in the input
            # `x` at index 1 before the time-enc dims were appended.
            # Fetch it so we can condition the flow on the per-observation error.
            meas_err_t = x[:, ti, 1]

            # If a flow head is configured, condition on [h_bi, meas_err] and
            # produce a point prediction based on flow_mode. Otherwise,
            # use the Gaussian MLP head as a fallback for compatibility.
            if self.flow is not None:
                ctx = torch.cat([h_bi, meas_err_t.unsqueeze(1)], dim=1)
                # zuko flow objects are callable and return a Distribution
                dist = self.flow(ctx)

                if flow_mode == 'sample':
                    # Single random sample (stochastic)
                    s = dist.sample()
                    point_prediction = s.view(B, 1)
                elif flow_mode == 'mode':
                    # Gradient-based optimization to find the mode (MAP estimate)
                    # We need to detach and rebuild the distribution to avoid graph issues
                    ctx_detached = ctx.detach()
                    dist_opt = self.flow(ctx_detached)
                    # Initialize at a sample from the distribution (better than zero)
                    y_init = dist_opt.sample().clone().detach()
                    y_opt = y_init.requires_grad_(True)
                    # Use Adam with smaller lr for more stable optimization
                    optimizer = torch.optim.Adam([y_opt], lr=0.1)
                    for _ in range(30):
                        optimizer.zero_grad()
                        # Rebuild distribution each step to get fresh graph
                        dist_step = self.flow(ctx_detached)
                        loss = -dist_step.log_prob(y_opt).sum()
                        loss.backward()
                        optimizer.step()
                        # Clamp to reasonable range to prevent divergence
                        with torch.no_grad():
                            y_opt.clamp_(-10, 10)
                    point_prediction = y_opt.detach().view(B, 1)
                else:  # 'mean' (default)
                    # Monte Carlo average of N samples to approximate the conditional mean
                    n_samples = 16
                    s_acc = dist.sample()
                    for _ in range(n_samples - 1):
                        s_acc = s_acc + dist.sample()
                    s_mean = s_acc / n_samples
                    point_prediction = s_mean.view(B, 1)
            else:
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
