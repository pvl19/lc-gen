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

from lcgen.models.MetadataAgePredictor import (
    MetadataEncoder, DEFAULT_METADATA_FIELDS,
    ASTRO_METADATA_FIELDS, INSTRUMENTAL_METADATA_FIELDS,
)

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

    @staticmethod
    def g(x):
        return torch.where(x >= 0, x+0.5, torch.sigmoid(x))

    @staticmethod
    def log_g(x):
        # log of g(x) where g(x) = relu(x)+0.5 for x>=0 and sigmoid(x) for x<0.
        # log(sigmoid(x)) = -softplus(-x). The two branches are continuous at x=0
        # (both give log(0.5) ≈ -0.693).
        return torch.where(x >= 0, (F.relu(x)+0.5).log(), -F.softplus(-x))

    @staticmethod
    def parallel_scan_log(log_coeffs, log_values):
        # log_coeffs: (B, T, H)   — log(1 - sigmoid(k)), the forget factor in log space
        # log_values: (B, T+1, H) — [log_g(h_0), log_z + log_g(h_tilde)]
        # Two single PyTorch ops with fast backward (vs. a T-step Python loop).
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h)[:, 1:]

    def step(self, x_t, h_prev=None, mask_t=None):
        if h_prev is None:
            h_prev = x_t.new_zeros(x_t.shape[0], self.W_h.out_features)
        z = torch.sigmoid(self.W_z(x_t))
        h_tilde = torch.tanh(self.W_h(x_t))
        h = (1.0 - z) * h_prev + z * h_tilde
        if mask_t is not None:
            # Hard recurrence gating: at gated steps (mask_t == 0) force the
            # update gate to 0 so h = h_prev — the masked/padded step passes
            # the state through unchanged and contributes nothing.
            keep = (mask_t > 0.5).unsqueeze(-1)  # (B, 1)
            h = torch.where(keep, h, h_prev)
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
    
    def step_parallel(self, x, h_0, mask=None):
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)
        # mask: (batch_size, seq_len) optional, 1 = keep, 0 = gated. At gated
        #   steps the update gate is forced to 0 (log_z -> large negative so the
        #   value contributes 0; log_coeffs -> 0 so the state is carried), giving
        #   h_t = h_{t-1}. The gated step contributes nothing to the recurrence
        #   and receives ~zero gradient.
        #
        # Uses the log-domain parallel scan: two cumulative ops (cumsum +
        # logcumsumexp) instead of a T-step Python loop. The loop creates T
        # nodes in the autograd graph, making backward ~80× slower than
        # forward. The log-domain version has O(1) graph depth regardless of T.
        #
        # Uses g(x) = relu(x)+0.5 (always ≥ 0.5, as in the original minGRU
        # paper) instead of tanh, because log_g requires non-negative inputs.
        k = self.W_z(x)                                               # (B, T, H)
        log_z      = -F.softplus(-k)                                  # log sigmoid(k)
        log_coeffs = -F.softplus(k)                                   # log(1 - sigmoid(k))
        if mask is not None:
            # Hard recurrence gating. Use a large finite negative (not -inf) so
            # all downstream arithmetic stays finite: exp(-1e9) underflows to 0.
            keep = (mask > 0.5).unsqueeze(-1)                         # (B, T, 1)
            log_z      = torch.where(keep, log_z, log_z.new_full((), -1e9))
            log_coeffs = torch.where(keep, log_coeffs, log_coeffs.new_zeros(()))
        # Treat h_0 as the literal initial hidden state (not a pre-activation),
        # matching the sequential `step` convention. h_0 is expected to be >= 0
        # (typically zeros). log(0) = -inf contributes 0 to logsumexp, which is
        # exactly what we want for a zero initial state.
        log_h_0    = h_0.clamp_min(torch.finfo(h_0.dtype).tiny).log() # (B, 1, H)
        log_h_tilde = minGRUCell.log_g(self.W_h(x))                  # (B, T, H)
        return minGRUCell.parallel_scan_log(
            log_coeffs, torch.cat([log_h_0, log_z + log_h_tilde], dim=1)
        )
        

class _GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer (DANN, Ganin & Lempitsky 2015).

    Identity on the forward pass; multiplies the gradient by (-lambda) on the
    backward pass. Placed between a feature and an adversary classifier so that
    minimising the adversary's loss *maximises* it w.r.t. the feature producer.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_: float = 1.0):
    """Apply the gradient-reversal layer with coefficient ``lambda_``."""
    return _GradReverse.apply(x, lambda_)


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
        meta_use_mask: bool = False,
        split_meta_encoders: bool = False,
        instr_emb_dim: int = 16,
        adversarial_sector_head: bool = False,
        num_sectors: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_flow = use_flow
        self.direction = direction
        self.mode = mode
        self.use_conv_channels = use_conv_channels
        # When True the metadata encoder takes an explicit binary validity mask
        # channel (input dim doubles, 13 -> 26). Required for DOROTHY-style
        # metadata masking; old checkpoints were trained with False.
        self.meta_use_mask = meta_use_mask
        # split_meta_encoders: route instrumental fields (sector/camera/ccd) to a
        # separate encoder that conditions only the head, not the RNN hidden
        # states. adversarial_sector_head: GRL adversary that scrubs sector from
        # the pooled latent. See docs/plans/2026-05-18_split-metadata-encoders.md.
        self.split_meta_encoders = split_meta_encoders
        self.adversarial_sector_head = adversarial_sector_head
        self.num_sectors = num_sectors

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
        self.instr_meta_encoder = None
        self.instr_emb_dim = 0
        self.astro_meta_idx = None
        self.instr_meta_idx = None
        if num_meta_features > 0:
            meta_emb_dim = 32
            if split_meta_encoders:
                # Column indices into the full metadata vector for each group.
                self.astro_meta_idx = [DEFAULT_METADATA_FIELDS.index(f)
                                       for f in ASTRO_METADATA_FIELDS]
                self.instr_meta_idx = [DEFAULT_METADATA_FIELDS.index(f)
                                       for f in INSTRUMENTAL_METADATA_FIELDS]
                # Astro encoder -> RNN hidden states (maskable, DOROTHY-style).
                self.meta_encoder = MetadataEncoder(
                    input_dim=len(self.astro_meta_idx),
                    latent_dim=meta_emb_dim,
                    hidden_dims=[128, 128, 128],
                    dropout=meta_dropout,
                    use_mask=meta_use_mask,
                )
                # Instrumental encoder -> prediction head only. Left UNMASKED:
                # it is out of the latent by construction and the head should
                # reliably see sector/camera/ccd.
                self.instr_emb_dim = instr_emb_dim
                self.instr_meta_encoder = MetadataEncoder(
                    input_dim=len(self.instr_meta_idx),
                    latent_dim=instr_emb_dim,
                    hidden_dims=[64, 64],
                    dropout=meta_dropout,
                    use_mask=False,
                )
            else:
                self.meta_encoder = MetadataEncoder(
                    input_dim=num_meta_features,
                    latent_dim=meta_emb_dim,
                    hidden_dims=[128, 128, 128],
                    dropout=meta_dropout,
                    use_mask=meta_use_mask,
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

        # With split metadata encoders the instrumental embedding is concatenated
        # into the head input (and only there), so both heads widen by instr_emb_dim.
        head_extra = self.instr_emb_dim
        self.gauss_head = nn.Sequential(
            nn.Linear(self.output_size + head_extra, head_hidden),
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
            # context dim = head input after head_norm/time-scaling (+ instr emb) + 1 for meas_err
            flow_context_dim = self.output_size + head_extra + 1
            # small NSF: 2 transforms, small hidden networks
            self.flow = zuko.flows.NSF(1, flow_context_dim, transforms=2, hidden_features=[64, 64])

        # Adversarial sector head (DANN). A small MLP predicts sector identity
        # from the masked mean+std pool of the hidden states, through a gradient
        # reversal layer — so the encoder is pushed to make the pooled latent
        # NOT sector-predictable. Train-only; discarded at inference.
        self.sector_adversary = None
        if adversarial_sector_head:
            if num_sectors <= 0:
                raise ValueError('adversarial_sector_head=True requires num_sectors > 0')
            n_dir = 2 if direction == 'bi' else 1
            adv_in = 2 * n_dir * hidden_size  # mean+std per direction
            self.sector_adversary = nn.Sequential(
                nn.Linear(adv_in, 128),
                nn.GELU(),
                nn.Linear(128, num_sectors),
            )

    def forward(self, x, t, mask=None, metadata=None, meta_mask=None,
                meta_block_drop=None, conv_data=None, return_states: bool = False,
                flow_mode: str = 'mean', adv_lambda: float = 0.0):
        """Forward pass through the model.

        Args:
            x: (B, L, 2) input tensor [flux, flux_err]
            t: (B, L) or (B, L, 1) timestamps
            mask: (B, L) optional sequence mask (1=observed, 0=masked/padded).
                  Masked positions have their flux/flux_err zeroed in the RNN
                  input AND are gated out of the minGRU recurrence.
            metadata: (B, num_meta_features) optional stellar metadata tensor.
                      If None, the metadata path is skipped.
            meta_mask: (B, num_meta_features) optional binary validity mask for
                      metadata fields (1=present, 0=masked). Used only when the
                      metadata encoder was built with meta_use_mask=True; None
                      -> all fields present.
            meta_block_drop: (B,) optional bool tensor; where True the star's
                      metadata embedding is zeroed (whole-encoder block drop).
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

        # Preserve the original (pre-mask) flux_err so the reconstruction head
        # can always condition the flow on the real measurement error, even at
        # masked positions. This matches the training-time loss, which uses
        # the unmasked flux_err in the flow context.
        flux_err_unmasked = x[..., 1].clone()  # (B, L)

        # Zero flux/flux_err at masked positions. This is now redundant with the
        # hard recurrence gating applied in the minGRU scans (gated steps
        # contribute nothing regardless of their input), but is kept as a cheap
        # belt-and-suspenders so masked inputs never leak even if gating is
        # bypassed. The real masking mechanism is the gating in step_parallel.
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
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
        # With split_meta_encoders, only the astrophysical fields feed the RNN;
        # the instrumental embedding (instr_emb) is computed here but used only
        # in the prediction head, never in the hidden states.
        instr_emb = None
        if self.meta_encoder is not None and metadata is not None:
            if self.split_meta_encoders:
                astro_meta = metadata[:, self.astro_meta_idx]
                astro_mask = (meta_mask[:, self.astro_meta_idx]
                              if meta_mask is not None else None)
                meta_emb = self.meta_encoder(astro_meta, astro_mask)
                # Instrumental encoder is unmasked by design.
                instr_emb = self.instr_meta_encoder(metadata[:, self.instr_meta_idx])
            else:
                # MetadataEncoder.forward ignores `mask` when built with
                # use_mask=False, so passing meta_mask unconditionally is safe.
                meta_emb = self.meta_encoder(metadata, meta_mask)  # (B, meta_emb_dim)
            if meta_block_drop is not None:
                # Whole-encoder block drop: zero the (astro) metadata embedding
                # for the selected stars so they receive no metadata context.
                keep = (~meta_block_drop).to(meta_emb.dtype).unsqueeze(-1)  # (B, 1)
                meta_emb = meta_emb * keep
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

        # ---- Store backward hidden states (recurrence gated on `mask`) ----
        # When `mask` is given, masked/padded steps (mask == 0) are gated out of
        # the minGRU recurrence so they carry the state unchanged — they do not
        # perturb the hidden state and do not leak into neighbouring positions.

        if self.direction in ['bi', 'backward']:
            h_bwd_tensor = x.new_zeros(B, L, self.hidden_size)
            h_bwd = x.new_zeros(B, self.hidden_size)
            # Backward pass reverses the sequence — the gating mask must reverse too.
            mask_bwd = mask.flip(dims=[1]) if mask is not None else None

            if self.mode == 'parallel':
                inp_bwd = x.flip(dims=[1])  # reverse sequence for backward pass
                x_bwd_proj = self.backward_input_proj(inp_bwd)
                h_bwd_all = self.backward_cell.step_parallel(x_bwd_proj, h_bwd.unsqueeze(1), mask=mask_bwd)
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
                    mask_ti = mask[:, ti] if mask is not None else None
                    h_bwd = self.backward_cell.step(inp_bwd, h_bwd, mask_t=mask_ti)

        if self.direction in ['bi', 'forward']:
            h_fwd_tensor = x.new_zeros(B, L, self.hidden_size)
            h_fwd = x.new_zeros(B, self.hidden_size)

            if self.mode == 'parallel':
                inp_fwd = x
                x_fwd_proj = self.forward_input_proj(inp_fwd)
                h_fwd_all = self.forward_cell.step_parallel(x_fwd_proj, h_fwd.unsqueeze(1), mask=mask)
                h_fwd_tensor = h_fwd_all
                h0_f = h_fwd_tensor.new_zeros(B, 1, self.hidden_size)
                h_fwd_tensor = torch.cat([h0_f, h_fwd_tensor[:, :-1, :]], dim=1)

            elif self.mode == 'sequential':
                for ti in range(L):
                    h_fwd_tensor[:, ti, :] = h_fwd

                    # Forward RNN step
                    xi_fwd = x[:, ti, :]
                    inp_fwd = self.forward_input_proj(xi_fwd)
                    mask_ti = mask[:, ti] if mask is not None else None
                    h_fwd = self.forward_cell.step(inp_fwd, h_fwd, mask_t=mask_ti)

        out = {'reconstructed': None}
        if not return_states:
            # Reconstruction pass: compute per-timestep predictions for visualization.
            seq_prediction = []
            for ti in range(L):
                time_enc_t = t_enc[:, ti, :]
                if self.direction == 'forward':
                    h_bi = torch.cat([h_fwd_tensor[:, ti, :], time_enc_t], dim=1)
                elif self.direction == 'backward':
                    h_bi = torch.cat([h_bwd_tensor[:, ti, :], time_enc_t], dim=1)
                else:
                    h_bi = torch.cat([h_fwd_tensor[:, ti, :], h_bwd_tensor[:, ti, :], time_enc_t], dim=1)
                h_bi = self.head_norm(h_bi)
                nt = self.num_time_enc_dims
                if nt > 0:
                    h_hidden = h_bi[:, :-nt]
                    h_time = h_bi[:, -nt:] * self.time_scale
                    h_bi = torch.cat([h_hidden, h_time], dim=1)
                # Split-encoder: instrumental embedding conditions the head only.
                if instr_emb is not None:
                    h_bi = torch.cat([h_bi, instr_emb], dim=1)
                meas_err_t = flux_err_unmasked[:, ti]
                if self.flow is not None:
                    ctx = torch.cat([h_bi, meas_err_t.unsqueeze(1)], dim=1)
                    dist = self.flow(ctx)
                    if flow_mode == 'sample':
                        point_prediction = dist.sample().view(B, 1)
                    elif flow_mode == 'mode':
                        ctx_detached = ctx.detach()
                        dist_opt = self.flow(ctx_detached)
                        y_init = dist_opt.sample().clone().detach()
                        y_opt = y_init.requires_grad_(True)
                        optimizer = torch.optim.Adam([y_opt], lr=0.1)
                        for _ in range(30):
                            optimizer.zero_grad()
                            dist_step = self.flow(ctx_detached)
                            loss = -dist_step.log_prob(y_opt).sum()
                            loss.backward()
                            optimizer.step()
                            with torch.no_grad():
                                y_opt.clamp_(-10, 10)
                        point_prediction = y_opt.detach().view(B, 1)
                    else:  # 'mean'
                        n_samples = 16
                        s_acc = dist.sample()
                        for _ in range(n_samples - 1):
                            s_acc = s_acc + dist.sample()
                        point_prediction = (s_acc / n_samples).view(B, 1)
                else:
                    point_prediction = self.gauss_head(h_bi)
                seq_prediction.append(point_prediction)
            out['reconstructed'] = torch.stack(seq_prediction, dim=1)  # (B, L, 1)

        if return_states:
            # Provide forward hidden states and time encodings so training code can
            # compute multi-step losses based on the hidden states BEFORE each timestep.
            if self.direction in ['bi', 'forward']:
                out['h_fwd_tensor'] = h_fwd_tensor
            if self.direction in ['bi', 'backward']:
                out['h_bwd_tensor'] = h_bwd_tensor
            out['t_enc'] = t_enc
            # Instrumental embedding: the loss concatenates it into the head
            # context (the only place split-encoder instrumental info is used).
            out['instr_emb'] = instr_emb
            # Adversarial sector head — train-only. Pool the hidden states, pass
            # through the gradient-reversal layer, predict sector.
            if self.sector_adversary is not None and self.training:
                pooled = self._pool_for_adversary(
                    out.get('h_fwd_tensor'), out.get('h_bwd_tensor'), mask)
                out['sector_logits'] = self.sector_adversary(
                    grad_reverse(pooled, adv_lambda))
        return out

    def _pool_for_adversary(self, h_fwd, h_bwd, mask):
        """Masked mean+std pool of the hidden states, per direction → (B, 2*n_dir*H).

        This is the per-star summary the adversary tries to predict sector from;
        scrubbing it pushes the encoder toward sector-free hidden states.
        """
        ref = h_fwd if h_fwd is not None else h_bwd
        B, L, H = ref.shape
        if mask is None:
            w = ref.new_ones(B, L, 1)
        else:
            w = (mask > 0.5).to(ref.dtype).unsqueeze(-1)  # (B, L, 1)
        denom = w.sum(dim=1).clamp_min(1.0)               # (B, 1)
        parts = []
        for h in (h_fwd, h_bwd):
            if h is None:
                continue
            mean = (h * w).sum(dim=1) / denom
            var = (((h - mean.unsqueeze(1)) ** 2) * w).sum(dim=1) / denom
            parts.append(mean)
            parts.append(var.clamp_min(1e-8).sqrt())
        return torch.cat(parts, dim=-1)

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
