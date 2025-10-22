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
from typing import List, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import ModelConfig, BaseAutoencoder


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
    input_length: int = 512

    # Architecture
    encoder_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    rnn_type: Literal["minlstm", "minGRU"] = "minGRU"  # minGRU is simpler and often performs better
    num_layers_per_level: int = 2  # RNN layers per hierarchy level
    bidirectional: bool = True  # Bidirectional RNN for reconstruction
    dropout: float = 0.0

    # Positional encoding (time-aware)
    min_period: float = 0.00278  # ~4 minutes in days
    max_period: float = 1640.0   # ~4.5 years in days
    num_freqs: int = 32


class TimeAwarePositionalEncoding(nn.Module):
    """
    Time-aware positional encoding using actual timestamps.

    Uses sinusoidal encoding with frequencies covering astrophysical timescales.
    """
    def __init__(self, d_model: int, min_period: float = 0.00278,
                 max_period: float = 1640.0, num_freqs: int = 32):
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


class StackedRNN(nn.Module):
    """Stack multiple RNN layers with residual connections and bidirectional support"""
    def __init__(self, rnn_type: str, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float = 0.0, bidirectional: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Choose RNN cell type
        RNNCell = minLSTMCell if rnn_type == "minlstm" else minGRUCell

        # Forward layers
        self.forward_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.forward_layers.append(RNNCell(layer_input_size, hidden_size))

        # Backward layers (if bidirectional)
        if bidirectional:
            self.backward_layers = nn.ModuleList()
            for i in range(num_layers):
                layer_input_size = input_size if i == 0 else hidden_size
                self.backward_layers.append(RNNCell(layer_input_size, hidden_size))

            # Fusion layer to combine forward and backward
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
        # Forward pass
        h_fwd = x
        for i, layer in enumerate(self.forward_layers):
            h_new = layer(h_fwd)

            # Residual connection (if dimensions match)
            if i > 0 and h_new.shape == h_fwd.shape:
                h_new = h_new + h_fwd

            h_fwd = self.dropout(self.norm(h_new))

        if not self.bidirectional:
            return h_fwd

        # Backward pass (process reversed sequence)
        h_bwd = x.flip(dims=[1])  # Reverse time dimension
        for i, layer in enumerate(self.backward_layers):
            h_new = layer(h_bwd)

            # Residual connection
            if i > 0 and h_new.shape == h_bwd.shape:
                h_new = h_new + h_bwd

            h_bwd = self.dropout(self.norm(h_new))

        h_bwd = h_bwd.flip(dims=[1])  # Flip back to original order

        # Concatenate and fuse
        h_combined = torch.cat([h_fwd, h_bwd], dim=-1)
        h = self.fusion(h_combined)

        return h


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
                    bidirectional=config.bidirectional
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
            bidirectional=config.bidirectional
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
                    bidirectional=config.bidirectional
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

        # Output projection
        self.output_proj = nn.Linear(config.encoder_dims[0], 1)

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
            (batch, seq_len, 1)
        """
        x = self.decoder(latent, skip_connections)
        x = F.gelu(x)  # Non-linearity before output projection
        x = self.output_proj(x)
        return x

    def forward(self, x, t=None):
        """
        Full forward pass.

        Args:
            x: (batch, seq_len, input_dim) where input_dim = [flux, mask]
            t: (batch, seq_len) timestamps (optional, will use indices if None)
        Returns:
            dict with 'reconstructed', 'latent', 'skip_connections'
        """
        batch_size, seq_len, _ = x.shape

        # Generate timestamps if not provided
        if t is None:
            t = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1).float()

        # Encode
        latent, skip_connections = self.encode(x, t)

        # Decode
        reconstructed = self.decode(latent, skip_connections)

        return {
            'reconstructed': reconstructed,
            'latent': latent,
            'skip_connections': skip_connections
        }


# Alias for consistency
TimeSeriesRNN = HierarchicalRNN
