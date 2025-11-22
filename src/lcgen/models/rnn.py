"""
Hierarchical RNN Autoencoder for Time Series Reconstruction

Implements a U-Net-style hierarchical RNN using minLSTM or minGRU cells
from "Were RNNs All We Needed?" (arXiv:2410.01201).

Key features:
- Parallelizable RNN variants (minGRU/minLSTM) using parallel scan algorithm
- Multi-scale temporal processing (512->256->128->64->32)
- Bidirectional processing for reconstruction tasks
- Skip connections between encoder and decoder
- Compatible with dynamic block masking
- Time-aware positional encoding
- 2× fewer parameters than Transformer (~13.4M vs ~26.8M)

Architecture:
    Input (seq_len, channels) -> Embedding
    -> Encoder Level 0-4 (RNN + downsample + skip)
    -> Bottleneck (32, hidden_dim)
    -> Decoder Level 4-0 (upsample + skip fusion + RNN)
    -> Output (seq_len, 1)
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import ModelConfig, BaseAutoencoder
# zuko is required for normalizing-flow posterior
try:
    import zuko
except Exception as e:
    raise ImportError(
        "The 'zuko' package is required for the normalizing-flow posterior."
        " Install it with: pip install zuko"
    ) from e


def parallel_scan_log(coeffs, values):
    """
    Memory-efficient associative scan for linear recurrence.

    Computes: h_t = coeffs_t * h_{t-1} + values_t

    Uses associative scan algorithm with O(log n) parallel depth and O(n) memory.

    Args:
        coeffs: Multiplicative coefficients (batch, seq_len, hidden)
        values: Additive values (batch, seq_len, hidden)

    Returns:
        h: Hidden states (batch, seq_len, hidden)

    Algorithm:
        Linear recurrence h_t = a_t * h_{t-1} + b_t can be represented as
        composition of affine functions: (a, b) represents x -> a*x + b

        The composition operator is associative:
        (a2, b2) ∘ (a1, b1) = (a2*a1, a2*b1 + b2)

        We use parallel prefix sum (scan) with this operator in O(log n) steps.

    Memory: O(batch * seq_len * hidden) - same as input
    Time: O(seq_len * log(seq_len)) - much better than O(seq_len^2)
    """
    batch_size, seq_len, hidden_size = coeffs.shape

    # Clamp coefficients for numerical stability
    a = torch.clamp(coeffs, min=1e-8, max=1.0 - 1e-8)
    b = values

    # Parallel associative scan using Blelloch-style algorithm
    # We do log2(seq_len) iterations, combining elements at power-of-2 distances

    # Number of parallel scan steps needed
    num_steps = (seq_len - 1).bit_length()  # ceil(log2(seq_len))

    for step in range(num_steps):
        distance = 1 << step  # 1, 2, 4, 8, 16, ...

        if distance >= seq_len:
            break

        # For each position t, combine with position (t - distance)
        # New: (a[t], b[t]) = (a[t], b[t]) ∘ (a[t-distance], b[t-distance])
        #                   = (a[t] * a[t-distance], a[t] * b[t-distance] + b[t])

        # Get elements at distance positions
        a_prev = F.pad(a[:, :-distance, :], (0, 0, distance, 0), value=1.0)  # Pad with 1s (identity for mult)
        b_prev = F.pad(b[:, :-distance, :], (0, 0, distance, 0), value=0.0)  # Pad with 0s (identity for add)

        # Compose: (a[t], b[t]) ∘ (a[t-distance], b[t-distance])
        a_new = a * a_prev
        b_new = a * b_prev + b

        # Update in-place for next iteration
        a = a_new
        b = b_new

    # After all iterations, b contains the final hidden states
    return b


@dataclass
class RNNConfig(ModelConfig):
    """Configuration for hierarchical RNN autoencoder"""
    model_type: str = "rnn"
    model_name: str = "hierarchical_rnn"

    # Input configuration
    input_dim: int = 3  # flux + flux_err + mask
    input_length: int = 1024

    # Architecture
    encoder_dims: List[int] = field(default_factory=lambda: [64, 128])
    rnn_type: Literal["minlstm", "minGRU"] = "minGRU"  # minGRU is simpler and often performs better
    num_layers_per_level: int = 2  # RNN layers per hierarchy level
    bidirectional: bool = False  # legacy boolean; kept for backward compatibility
    # Direction mode: 'forward' | 'backward' | 'both'. If set, overrides `bidirectional`.
    direction: Literal['forward', 'backward', 'both'] = 'forward'
    dropout: float = 0.0

    # Autoregressive generation
    autoregressive: bool = True
    teacher_forcing: bool = True

    # Positional encoding (time-aware)
    min_period: float = 0.00278  # ~4 minutes in days
    max_period: float = 28   # ~4.5 years in days
    num_freqs: int = 16
    # Variance propagation weight (0..1) when autoregressive; how much previous variance contributes
    var_propagation_alpha: float = 0.5


class TimeAwarePositionalEncoding(nn.Module):
    """
    Time-aware positional encoding using actual timestamps.

    Uses sinusoidal encoding with frequencies covering astrophysical timescales.
    """
    def __init__(self, d_model: int, min_period: float = 0.00278,
                 max_period: float = 28, num_freqs: int = 16):
        super().__init__()
        self.d_model = d_model

        # Log-spaced frequencies covering min to max period
        periods = torch.logspace(
            math.log10(min_period),
            math.log10(max_period),
            num_freqs
        )
        freqs = 1.0 / periods
        self.register_buffer('freqs', freqs)

        # Project encoding to d_model dimensions
        self.projection = nn.Linear(2 * num_freqs, d_model)

    def forward(self, t):
        """
        Args:
            t: Timestamps of shape (batch, seq_len)
        Returns:
            Positional encoding of shape (batch, seq_len, d_model)
        """
        # Compute phases: (batch, seq_len, num_freqs)
        phases = 2 * math.pi * t.unsqueeze(-1) * self.freqs

        # Sin and cos features: (batch, seq_len, 2*num_freqs)
        encoding = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)

        # Project to d_model: (batch, seq_len, d_model)
        return self.projection(encoding)


class minLSTMCell(nn.Module):
    """
    Minimal LSTM cell from "Were RNNs All We Needed?" (arXiv:2410.01201)

    Key differences from standard LSTM:
    - Gates don't depend on previous hidden state h_{t-1}
    - Fully parallelizable over sequence length using parallel scan
    - Much faster than traditional LSTM

    Linear recurrence: c_t = f_t * c_{t-1} + (1 - f_t) * i_t
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate (only depends on input)
        self.W_f = nn.Linear(input_size, hidden_size)

        # Input modulation (candidate values)
        self.W_i = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        """
        Process entire sequence in parallel using parallel scan.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        Returns:
            h: Hidden states of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Compute gates for entire sequence (fully parallel!)
        f = torch.sigmoid(self.W_f(x))  # (batch, seq_len, hidden)
        i = self.W_i(x)                  # (batch, seq_len, hidden)

        # Linear recurrence: c_t = f_t * c_{t-1} + (1 - f_t) * i_t
        # This can be parallelized using associative scan
        c = parallel_scan_log(f, (1 - f) * i)

        return c

    def step(self, x_t, c_prev=None):
        """Single-step update for minLSTMCell.

        x_t: (batch, input_size)
        c_prev: (batch, hidden_size) or None -> treated as zeros
        returns c: (batch, hidden_size)
        """
        # If previous cell state not provided, assume zeros
        if c_prev is None:
            batch_size = x_t.shape[0]
            c_prev = x_t.new_zeros(batch_size, self.hidden_size)

        f_t = torch.sigmoid(self.W_f(x_t))
        i_t = self.W_i(x_t)
        c = f_t * c_prev + (1 - f_t) * i_t
        return c


class minGRUCell(nn.Module):
    """
    Minimal GRU cell from "Were RNNs All We Needed?" (arXiv:2410.01201)

    Even simpler than minLSTM, competitive performance.

    Linear recurrence: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate
        self.W_z = nn.Linear(input_size, hidden_size)

        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        """
        Process entire sequence in parallel using parallel scan.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        Returns:
            h: Hidden states of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Compute for entire sequence (fully parallel!)
        z = torch.sigmoid(self.W_z(x))  # (batch, seq_len, hidden)
        h_tilde = torch.tanh(self.W_h(x))  # (batch, seq_len, hidden)

        # Linear recurrence: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
        # This can be parallelized using associative scan
        h = parallel_scan_log(1 - z, z * h_tilde)

        return h

    def step(self, x_t, h_prev=None):
        """Single-step update for minGRUCell.

        x_t: (batch, input_size)
        h_prev: (batch, hidden_size) or None -> treated as zeros
        returns h: (batch, hidden_size)
        """
        if h_prev is None:
            batch_size = x_t.shape[0]
            h_prev = x_t.new_zeros(batch_size, self.hidden_size)

        z = torch.sigmoid(self.W_z(x_t))
        h_tilde = torch.tanh(self.W_h(x_t))
        h = (1.0 - z) * h_prev + z * h_tilde
        return h


class StackedRNN(nn.Module):
    """Stack multiple RNN layers with residual connections and flexible direction modes.

    Supports three modes controlled by `direction`:
      - 'forward' (default): run forward-only layers
      - 'backward': run layers on reversed time sequence
      - 'both': run forward and backward, then fuse (learned linear) the two outputs

    For backward compatibility the `bidirectional` boolean is still accepted; when
    bidirectional=True it maps to direction='both'.
    """
    def __init__(self, rnn_type: str, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float = 0.0, bidirectional: bool = True,
                 direction: Optional[str] = None):
        super().__init__()

        # Resolve direction: explicit `direction` overrides `bidirectional` flag.
        if direction is None:
            self.direction = 'both' if bidirectional else 'forward'
        else:
            assert direction in ('forward', 'backward', 'both'), "direction must be 'forward','backward', or 'both'"
            self.direction = direction

        self.num_layers = num_layers

        # Choose RNN cell type
        RNNCell = minLSTMCell if rnn_type == "minlstm" else minGRUCell

        # Forward layers (always created)
        self.forward_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.forward_layers.append(RNNCell(layer_input_size, hidden_size))

        # Backward layers (created only if needed)
        if self.direction in ('backward', 'both'):
            self.backward_layers = nn.ModuleList()
            for i in range(num_layers):
                layer_input_size = input_size if i == 0 else hidden_size
                self.backward_layers.append(RNNCell(layer_input_size, hidden_size))

        # Fusion when bidirectional ('both') to map 2*hidden -> hidden
        if self.direction == 'both':
            self.fusion = nn.Linear(2 * hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, seq_len, hidden_size)
        """
        # Helper to run a sequence of layers (with residual behavior)
        def run_layers(layers, inp):
            h = inp
            for i, layer in enumerate(layers):
                h_new = layer(h)
                if i > 0 and h_new.shape == h.shape:
                    h_new = h_new + h
                h = self.dropout(self.norm(h_new))
            return h

        # Forward pass
        out_fwd = run_layers(self.forward_layers, x)

        if self.direction == 'forward':
            return out_fwd

        # Backward-only
        x_rev = x.flip(dims=[1])
        out_bwd = run_layers(self.backward_layers, x_rev)
        out_bwd = out_bwd.flip(dims=[1])

        if self.direction == 'backward':
            return out_bwd

        # Both: fuse forward+backward
        out_comb = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.fusion(out_comb)
        return out

    def step(self, x_t, h_prev=None):
        """Run a single timestep through the stacked RNN.

        Args:
            x_t: (batch, input_size) input at current timestep
            h_prev: optional list of previous hidden states per layer (not used by minimal cells)
        Returns:
            h_out: (batch, hidden_size) output for this timestep
        """
        # Run single-step through forward layers
        h = x_t
        for i, layer in enumerate(self.forward_layers):
            # each layer has .step
            h_prev_layer = None
            if hasattr(layer, 'step'):
                h = layer.step(h, h_prev_layer) if h.dim() == 2 else layer.step(h, None)
            else:
                # fallback - treat layer as stateless mapping
                h = layer(h.unsqueeze(1)).squeeze(1)
            # residual if shapes match and not first layer
            # here h is (batch, hidden)
        if self.direction == 'forward':
            return self.dropout(self.norm(h))

        # For backward or both modes we'll rely on sequence-level forward for now
        # since per-step backward generation is done by reversing input externally.
        # The `both` fusion happens at sequence level, so step() only returns forward output.
        return self.dropout(self.norm(h))


class Downsample1D(nn.Module):
    """Downsample sequence by factor of 2 using strided convolution"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_dim)  # Match Transformer

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            (batch, seq_len//2, out_channels)
        """
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len//2, out_channels)
        return self.norm(x)  # Normalize after downsampling


class Upsample1D(nn.Module):
    """Upsample sequence by factor of 2 using transposed convolution"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_dim)  # Match Transformer

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            (batch, seq_len*2, out_channels)
        """
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len*2, out_channels)
        return self.norm(x)  # Normalize after upsampling


class HierarchicalRNNEncoder(nn.Module):
    """
    Multi-scale RNN encoder with skip connections.

    Processes at resolutions: 512 -> 256 -> 128 -> 64 -> 32
    """
    def __init__(self, config: RNNConfig):
        super().__init__()
        self.num_levels = len(config.encoder_dims)

        # Build encoder levels
        self.encoder_levels = nn.ModuleList()

        for i in range(self.num_levels):
            # Match Transformer logic: process at in_dim, downsample to out_dim
            in_dim = config.encoder_dims[i]
            out_dim = config.encoder_dims[i+1] if i+1 < len(config.encoder_dims) else config.encoder_dims[-1]

            level = nn.ModuleDict({
                'rnn': StackedRNN(
                    rnn_type=config.rnn_type,
                    input_size=in_dim,
                    hidden_size=in_dim,  # Process at CURRENT resolution
                    num_layers=config.num_layers_per_level,
                    dropout=config.dropout,
                    bidirectional=config.bidirectional,
                    direction=config.direction
                ),
                'downsample': Downsample1D(in_dim, out_dim)
            })
            self.encoder_levels.append(level)

        # Bottleneck RNN
        bottleneck_dim = config.encoder_dims[-1]
        self.bottleneck = StackedRNN(
            rnn_type=config.rnn_type,
            input_size=bottleneck_dim,
            hidden_size=bottleneck_dim,
            num_layers=config.num_layers_per_level,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            direction=config.direction
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            bottleneck: (batch, seq_len//16, channels)
            skip_connections: List of (batch, seq_len_i, channels_i)
        """
        skip_connections = []

        # Encoder pathway
        for level in self.encoder_levels:
            # Process at current resolution
            x = level['rnn'](x)

            # Store skip connection BEFORE downsampling
            skip_connections.append(x.clone())

            # Downsample for next level
            x = level['downsample'](x)

        # Bottleneck
        x = self.bottleneck(x)

        return x, skip_connections


class HierarchicalRNNDecoder(nn.Module):
    """
    Multi-scale RNN decoder with skip connection fusion.

    Upsamples: 32 -> 64 -> 128 -> 256 -> 512
    """
    def __init__(self, config: RNNConfig):
        super().__init__()
        self.num_levels = len(config.encoder_dims)

        # Build decoder levels (reverse order) - match Transformer logic
        self.decoder_levels = nn.ModuleList()

        for i in reversed(range(self.num_levels)):
            # in_dim: dimension we're upsampling FROM
            # out_dim: dimension we're upsampling TO (matches skip connection)
            in_dim = config.encoder_dims[i+1] if i+1 < len(config.encoder_dims) else config.encoder_dims[-1]
            out_dim = config.encoder_dims[i]

            level = nn.ModuleDict({
                'upsample': Upsample1D(in_dim, out_dim),
                'fusion': nn.Linear(out_dim * 2, out_dim),  # Concatenate skip + upsampled
                'rnn': StackedRNN(
                    rnn_type=config.rnn_type,
                    input_size=out_dim,
                    hidden_size=out_dim,
                    num_layers=config.num_layers_per_level,
                    dropout=config.dropout,
                    bidirectional=config.bidirectional,
                    direction=config.direction
                )
            })
            self.decoder_levels.append(level)

    def forward(self, x, skip_connections):
        """
        Args:
            x: Bottleneck (batch, seq_len//16, channels)
            skip_connections: List from encoder (in forward order)
        Returns:
            (batch, seq_len, channels)
        """
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Decoder pathway
        for i, level in enumerate(self.decoder_levels):
            # Upsample
            x = level['upsample'](x)

            # Get corresponding skip connection
            skip = skip_connections[i]

            # Handle potential size mismatch from conv/deconv
            if x.shape[1] != skip.shape[1]:
                x = F.interpolate(
                    x.transpose(1, 2),
                    size=skip.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            # Concatenate and fuse
            x = torch.cat([x, skip], dim=-1)
            x = level['fusion'](x)

            # Process with RNN
            x = level['rnn'](x)

        return x


class HierarchicalRNN(BaseAutoencoder):
    """
    Hierarchical RNN Autoencoder using minLSTM or minGRU cells.

    Based on "Were RNNs All We Needed?" (arXiv:2410.01201)
    """
    def __init__(self, config: RNNConfig):
        super().__init__(config)
        self.config = config

        # Input embedding
        self.input_proj = nn.Linear(config.input_dim, config.encoder_dims[0])

        # Positional encoding
        self.pos_encoder = TimeAwarePositionalEncoding(
            d_model=config.encoder_dims[0],
            min_period=config.min_period,
            max_period=config.max_period,
            num_freqs=config.num_freqs
        )

        # Encoder and Decoder
        self.encoder = HierarchicalRNNEncoder(config)
        self.decoder = HierarchicalRNNDecoder(config)
        # Output projection: keep a small Gaussian head for compatibility
        # output channels: [mean, log_sigma]
        self.output_proj_gauss = nn.Linear(config.encoder_dims[0], 2)
        # Flow context projection: produces conditioning vector per timestep for the flow
        self.flow_context_proj = nn.Linear(config.encoder_dims[0], config.encoder_dims[0])
        # Conditional normalizing flow posterior (zuko). Lazy - may raise helpful error if missing at runtime.
        self.flow_posterior = None
        if zuko is not None:
            # create a small NSF for 1D data conditioned on flow_context_dim
            flow_context_dim = config.encoder_dims[0]
            # Use a small number of transforms for speed; user can tune later
            self.flow_posterior = zuko.flows.NSF(1, flow_context_dim, transforms=2, hidden_features=[64, 64])
        else:
            # Leave as None; code will raise a clear error if used without zuko installed
            self.flow_posterior = None
        # small projection to mix previous mean into step input
        self.prev_mean_proj = nn.Linear(1, config.encoder_dims[0])
        # small projection to inject per-timestep measurement error into step input
        self.meas_proj = nn.Linear(1, config.encoder_dims[0])
        # Stepwise stacked RNN used for autoregressive decoding (operates at base resolution)
        # Use same rnn_type and width as level 0
        self.step_rnn = StackedRNN(
            rnn_type=config.rnn_type,
            input_size=config.encoder_dims[0],
            hidden_size=config.encoder_dims[0],
            num_layers=config.num_layers_per_level,
            dropout=config.dropout,
            bidirectional=False,
            direction='forward'
        )

    def encode(self, x, t):
        """
        Encode time series to latent representation.

        Args:
            x: (batch, seq_len, input_dim)
            t: (batch, seq_len) timestamps
        Returns:
            latent: (batch, seq_len//16, hidden_dim)
            skip_connections: List of intermediate representations
        """
        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        pos_enc = self.pos_encoder(t)
        x = x + pos_enc

        # Encode
        latent, skip_connections = self.encoder(x)

        return latent, skip_connections

    def decode(self, latent, skip_connections):
        """
        Decode latent to reconstructed time series.

        Args:
            latent: (batch, seq_len//16, hidden_dim)
            skip_connections: List from encoder
        Returns:
            (batch, seq_len, 2)  # [mean, log_sigma]
        """
        x = self.decoder(latent, skip_connections)
        x = F.gelu(x)  # Non-linearity before output projections

        # Gaussian head (compatibility)
        gauss_out = self.output_proj_gauss(x)

        # Flow conditioning context per timestep
        flow_ctx = self.flow_context_proj(x)

        # Return both so callers can use the flow when available
        return gauss_out, flow_ctx

    def forward(self, x, t=None, target=None, mask=None, seed: 'Optional[int]' = None):
        """
        Full forward pass.

        Args:
            x: (batch, seq_len, input_dim) where input_dim = [flux, mask]
            t: (batch, seq_len) timestamps (optional, will use indices if None)
            Returns:
            dict with 'reconstructed' (batch, seq_len, 2 -> mean, log_sigma), 'latent', 'skip_connections'
        """
        batch_size, seq_len, _ = x.shape

        # Generate timestamps if not provided
        if t is None:
            t = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1).float()

        # Encode
        latent, skip_connections = self.encode(x, t)

        # If autoregressive mode requested, use sequential generation
        if getattr(self.config, 'autoregressive', False):
            ar_out = self.generate_autoregressive(x, t, target=target, mask=mask, seed=seed)
            # ar_out is a dict {'reconstructed': tensor, 'sampled': tensor or None}
            ar_out['latent'] = latent
            ar_out['skip_connections'] = skip_connections
            return ar_out
        else:
            gauss_out, flow_ctx = self.decode(latent, skip_connections)
            return {
                'reconstructed': gauss_out,
                'flow_context': flow_ctx,
                'latent': latent,
                'skip_connections': skip_connections
            }

    def generate_autoregressive(self, x, t, target=None, mask=None, seed: 'Optional[int]' = None):
        """Generate predictions sequentially using teacher forcing when available.

        Args:
            x: (batch, seq_len, 3) input channels [masked_flux, flux_err, mask]
            t: (batch, seq_len) timestamps
            target: optional (batch, seq_len) ground truth flux for teacher forcing
            mask: optional (batch, seq_len) boolean mask where True indicates masked positions

        Returns:
            reconstructed: (batch, seq_len, 2) mean and log_sigma
        """
        device = x.device
        batch_size, seq_len, _ = x.shape

        # Project input once and get encoder context (not strictly used here but kept)
        latent, skip_connections = self.encode(x, t)

        # Compute decoder features once: fuse latent + skip connections into
        # per-timestep context that includes positional and skip information.
        # Note: encode() already added positional encoding before the encoder,
        # and the decoder output therefore contains time-aware features.
        dec_feat = self.decoder(latent, skip_connections)  # (B, L, D)
        # Decoder-level flow context (fused across skips/latent)
        flow_ctx_dec = self.flow_context_proj(dec_feat)

        # helper: inverse softplus so we can convert sigma back to pre-activation
        def inv_softplus(y):
            # numerically stable inverse softplus: inv_sp(y) = log(exp(y) - 1)
            # handle large y with approximation
            thresh = 20.0
            y_clip = torch.clamp(y, max=thresh)
            out = torch.log(torch.expm1(y_clip).clamp_min(1e-12))
            out = torch.where(y > thresh, y - math.log(1.0), out)
            return out

        # Prepare a random generator for reproducible sampling when a seed is provided
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))

        def run_direction(forward=True):
            # returns means (B,L), sigmas (B,L), samples (B,L or None) for the direction
            if forward:
                indices = range(seq_len)
            else:
                indices = range(seq_len - 1, -1, -1)

            prev_mean = torch.zeros(batch_size, device=device)

            means = []
            sigmas = []
            samples = []  # store sampled values when sampling is active

            for ti in indices:
                # Use decoder features as the per-step static context
                step_feat = dec_feat[:, ti, :]

                # measurement error at this timestep (scalar and 1-dim for projection)
                meas_err_scalar = x[:, ti, 1]
                meas_err = x[:, ti, 1:2]

                # mix previous mean via a small projection
                prev_mean_emb = self.prev_mean_proj(prev_mean.unsqueeze(-1))
                # project measurement error and add it to step input so the
                # autoregressive step conditions explicitly on local uncertainty
                meas_emb = self.meas_proj(meas_err)

                step_in = step_feat + prev_mean_emb + meas_emb

                step_out = self.step_rnn.step(step_in)

                # Gaussian head for compatibility (kept)
                out2 = self.output_proj_gauss(step_out)
                mean_t = out2[:, 0]
                logsigma_raw = out2[:, 1]

                # Flow context for this timestep
                flow_ctx_t = self.flow_context_proj(step_out)  # (B, C)

                # keep scalar measurement error available for diagnostics / downstream use
                # (meas_err_scalar set earlier)
                meas_err = meas_err_scalar

                means.append(mean_t.unsqueeze(1))
                # store raw flow context (not a sigma)
                sigmas.append(logsigma_raw.unsqueeze(1))

                # Determine per-sample whether this timestep is masked (True => masked)
                if mask is not None:
                    mask_t = mask[:, ti].to(device).bool()
                else:
                    mask_t = torch.zeros(batch_size, dtype=torch.bool, device=device)

                # Decide per-sample whether to use teacher forcing:
                if (target is None) or (not getattr(self.config, 'teacher_forcing', True)):
                    use_teacher = torch.zeros(batch_size, dtype=torch.bool, device=device)
                else:
                    # Use teacher forcing when the timestep is NOT masked (observed)
                    use_teacher = (~mask_t)

                # Prepare container for values used as previous mean for next step
                sampled_vals = torch.empty(batch_size, device=device)

                # Fill in teacher-forced entries
                if use_teacher.any():
                    sampled_vals[use_teacher] = target[use_teacher, ti]

                # For remaining entries, sample from the conditional flow posterior if available
                non_teacher_idx = (~use_teacher)
                n_non_teacher = int(non_teacher_idx.sum().item())
                if n_non_teacher > 0:
                        if self.flow_posterior is None:
                            # fallback: sample from Gaussian head
                            eps = torch.randn(n_non_teacher, device=device, generator=gen) if gen is not None else torch.randn(n_non_teacher, device=device)
                            sampled_vals[non_teacher_idx] = mean_t[non_teacher_idx] + F.softplus(logsigma_raw[non_teacher_idx]) * eps
                        else:
                            # sample from flow conditioned on flow_ctx_t (step-level context)
                            # select contexts for non-teacher indices and sample
                            ctx_nt = flow_ctx_t[non_teacher_idx]
                            # zuko expects context shape (..., C) and returns a Distribution
                            dist = self.flow_posterior(ctx_nt)
                            # Try to pass generator to make sampling deterministic when available
                            try:
                                s = dist.sample(generator=gen) if gen is not None else dist.sample()
                            except TypeError:
                                # Some distribution implementations may not accept generator
                                s = dist.sample()
                            # squeeze event dim
                            s = s.view(-1)
                            sampled_vals[non_teacher_idx] = s.to(device)

                samples.append(sampled_vals.unsqueeze(1))
                prev_mean = sampled_vals

            if not forward:
                # reverse collected lists
                means = means[::-1]
                sigmas = sigmas[::-1]

            means = torch.cat(means, dim=1)
            sigmas = torch.cat(sigmas, dim=1)
            samples = torch.cat(samples, dim=1) if len(samples) > 0 else None
            return means, sigmas, samples

        # Run according to direction: 'forward', 'backward', or 'both'
        dir_mode = getattr(self.config, 'direction', 'forward')
        if dir_mode == 'forward':
            means_f, sig_raw_f, samples_f = run_direction(forward=True)
            recon = torch.stack([means_f, sig_raw_f], dim=-1)
            return {'reconstructed': recon, 'sampled': samples_f}
        elif dir_mode == 'backward':
            means_b, sig_raw_b, samples_b = run_direction(forward=False)
            recon = torch.stack([means_b, sig_raw_b], dim=-1)
            return {'reconstructed': recon, 'sampled': samples_b}
        else:
            # both: run forward and backward and fuse by precision weighting
            means_f, sig_raw_f, samples_f = run_direction(forward=True)
            means_b, sig_raw_b, samples_b = run_direction(forward=False)

            sigma_f = F.softplus(sig_raw_f)
            sigma_b = F.softplus(sig_raw_b)
            var_f = sigma_f ** 2 + 1e-12
            var_b = sigma_b ** 2 + 1e-12

            prec_f = 1.0 / var_f
            prec_b = 1.0 / var_b

            combined_prec = prec_f + prec_b
            combined_mean = (means_f * prec_f + means_b * prec_b) / combined_prec
            combined_var = 1.0 / combined_prec
            combined_sigma = torch.sqrt(combined_var)
            combined_raw = inv_softplus(combined_sigma)

            recon = torch.stack([combined_mean, combined_raw], dim=-1)

            # Prefer flow-resampling from the fused (decoder-level) context for
            # masked entries; keep teacher-forcing for observed points.
            samples_combined = None

            # Decide per-position whether to use teacher forcing (observed points)
            if (target is None) or (not getattr(self.config, 'teacher_forcing', True)):
                use_teacher_mask = torch.zeros_like(combined_mean, dtype=torch.bool, device=device)
            else:
                if mask is None:
                    use_teacher_mask = torch.zeros_like(combined_mean, dtype=torch.bool, device=device)
                else:
                    use_teacher_mask = (~mask).to(device).bool()

            # Initialize sampled tensor
            B, L = combined_mean.shape
            sampled_vals = torch.empty((B, L), device=device)

            # Fill teacher-forced entries with ground truth when available
            if use_teacher_mask.any():
                sampled_vals[use_teacher_mask] = target[use_teacher_mask]

            # Non-teacher positions should be sampled from the fused flow when available
            non_teacher_mask = ~use_teacher_mask
            if non_teacher_mask.any():
                if self.flow_posterior is not None:
                    # Use decoder-level flow context for fused sampling
                    ctx_flat = flow_ctx_dec.view(B * L, -1)
                    idx = non_teacher_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
                    if idx.numel() > 0:
                        ctx_sel = ctx_flat[idx]
                        dist = self.flow_posterior(ctx_sel)
                        try:
                            s = dist.sample(generator=gen) if gen is not None else dist.sample()
                        except TypeError:
                            s = dist.sample()
                        s = s.view(-1)
                        sampled_vals.view(-1)[idx] = s.to(device)
                else:
                    # Fallback: sample from combined Gaussian
                    eps = torch.randn_like(combined_mean, device=device, generator=gen) if gen is not None else torch.randn_like(combined_mean, device=device)
                    sampled_vals[non_teacher_mask] = combined_mean[non_teacher_mask] + combined_sigma[non_teacher_mask] * eps[non_teacher_mask]

            samples_combined = sampled_vals

            return {'reconstructed': recon, 'sampled': samples_combined}


# Alias for consistency
TimeSeriesRNN = HierarchicalRNN
