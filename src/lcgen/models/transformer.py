"""Hierarchical U-Net-style Transformer for time-series light curve reconstruction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List

from .base import BaseAutoencoder, ModelConfig


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for hierarchical Transformer autoencoder"""
    model_type: str = "transformer"
    model_name: str = "transformer_lc"

    # Architecture
    input_dim: int = 2  # flux + mask (explicit mask channel)
    input_length: int = 512  # Sequence length
    encoder_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_layers: int = 4  # Number of hierarchical levels
    nhead: int = 4  # Number of attention heads
    num_transformer_blocks: int = 2  # Transformer blocks per level
    dropout: float = 0.0  # No dropout, consistent with MLP/UNet

    # Positional encoding (for time-series)
    min_period: float = 0.00278  # Minimum period in days (~4 minutes)
    max_period: float = 1640.0   # Maximum period in days (~45 hours)

    def __post_init__(self):
        """Compute derived properties after initialization"""
        super().__post_init__()
        # Ensure num_layers matches encoder_dims
        if len(self.encoder_dims) != self.num_layers:
            self.num_layers = len(self.encoder_dims)

    @property
    def latent_dim(self) -> int:
        """Latent dimension is channels * spatial_size at bottleneck"""
        bottleneck_spatial_size = self.input_length // (2 ** self.num_layers)
        return self.encoder_dims[-1] * bottleneck_spatial_size


class TimeChannelPositionalEncoding(nn.Module):
    """
    Positional encoding based on actual time values (not indices).

    Uses log-spaced frequencies to encode time information across
    multiple scales, designed for astrophysical light curves.
    """

    def __init__(self, embed_dim: int, min_period: float, max_period: float):
        super().__init__()

        # Ensure embed_dim is even for sin/cos pairing
        assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"

        self.embed_dim = embed_dim
        self.min_period = min_period
        self.max_period = max_period

        # Precompute angular frequencies (log-spaced for multi-scale coverage)
        num_freqs = embed_dim // 2
        periods = torch.logspace(np.log10(min_period), np.log10(max_period), num_freqs)
        self.register_buffer("div_term", 2 * np.pi / periods)  # shape: (num_freqs,)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings (batch, seq_len, embed_dim)
            t: Time values (batch, seq_len) or (batch, seq_len, 1) - real timestamps

        Returns:
            x + positional encoding
        """
        batch_size, seq_len, _ = x.shape

        # Handle time dimension
        if t.dim() == 3:
            t = t.squeeze(-1)  # (batch, seq_len)

        # Use raw timestamps (sinusoidal encoding is scale-invariant)
        # Compute angles: (batch, seq_len, num_freqs)
        t_expanded = t.unsqueeze(-1)  # (batch, seq_len, 1)
        angles = t_expanded * self.div_term.unsqueeze(0).unsqueeze(0)

        # Sin/cos encoding: (batch, seq_len, embed_dim)
        pe = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device, dtype=x.dtype)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)

        return x + pe


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with multi-head attention.

    Uses pre-norm architecture (LayerNorm before attention/FFN) for better
    training stability.
    """

    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()

        # Pre-norm: LayerNorm before attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True  # (batch, seq, feature) format
        )

        # Pre-norm: LayerNorm before feedforward
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),  # GELU activation, consistent with MLP/UNet
            nn.Linear(ff_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm architecture.

        Args:
            x: Input (batch, seq_len, d_model)
            key_padding_mask: Optional padding mask (batch, seq_len)

        Returns:
            Output (batch, seq_len, d_model)
        """
        # Pre-norm multi-head attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_out)

        # Pre-norm feedforward with residual
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout(ff_out)

        return x


class Downsample1D(nn.Module):
    """Downsample sequence by factor of 2 using strided convolution"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            (batch, seq_len//2, out_channels)
        """
        # Transpose to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Transpose back to (batch, seq_len//2, channels)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class Upsample1D(nn.Module):
    """Upsample sequence by factor of 2 using transposed convolution"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            (batch, seq_len*2, out_channels)
        """
        # Transpose to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Transpose back to (batch, seq_len*2, channels)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class HierarchicalTransformerEncoder(nn.Module):
    """
    Hierarchical U-Net-style Transformer encoder.

    Progressive downsampling with Transformer blocks at each level.
    Stores skip connections for decoder.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        print(f"Building Hierarchical Transformer Encoder with {config.num_layers} levels")

        self.num_layers = config.num_layers
        self.encoder_dims = config.encoder_dims

        # Initial embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(config.input_dim, config.encoder_dims[0]),
            nn.LayerNorm(config.encoder_dims[0]),
            nn.GELU(),
        )

        # Positional encoding (optional, for time-series)
        self.pos_encoding = TimeChannelPositionalEncoding(
            config.encoder_dims[0],
            config.min_period,
            config.max_period
        )

        # Encoder levels with downsampling
        self.encoder_levels = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = config.encoder_dims[i]
            out_dim = config.encoder_dims[i+1] if i+1 < len(config.encoder_dims) else config.encoder_dims[-1]

            # Transformer blocks at CURRENT resolution (in_dim, before downsampling)
            transformer_blocks = nn.ModuleList([
                TransformerBlock(
                    d_model=in_dim,  # Process at current resolution
                    nhead=config.nhead,
                    ff_dim=in_dim * 4,  # Standard 4x expansion
                    dropout=config.dropout
                )
                for _ in range(config.num_transformer_blocks)
            ])

            # Downsampling layer (applies after transformer blocks)
            downsample = Downsample1D(in_dim, out_dim)

            self.encoder_levels.append(nn.ModuleDict({
                'downsample': downsample,
                'transformer_blocks': transformer_blocks
            }))

        # Bottleneck
        bottleneck_dim = config.encoder_dims[-1]
        self.bottleneck = nn.ModuleList([
            TransformerBlock(
                d_model=bottleneck_dim,
                nhead=config.nhead,
                ff_dim=bottleneck_dim * 4,
                dropout=config.dropout
            )
            for _ in range(config.num_transformer_blocks)
        ])

        # Calculate expected bottleneck size
        self.expected_bottleneck_length = config.input_length // (2 ** self.num_layers)
        print(f"Expected bottleneck spatial size: {self.expected_bottleneck_length}")

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through hierarchical encoder.

        Args:
            x: Input (batch, seq_len, input_dim) where input_dim = [flux, mask]
            t: Timestamps (batch, seq_len) - optional for positional encoding
            key_padding_mask: Optional padding mask (batch, seq_len)

        Returns:
            bottleneck_features: Encoded representation (batch, seq_len_small, d_model)
            skip_connections: List of skip connection tensors
        """
        # Initial embedding
        x = self.embedding(x)

        # Add positional encoding if timestamps provided
        if t is not None:
            x = self.pos_encoding(x, t)

        skip_connections = []

        # Progressive downsampling with Transformer blocks
        for i, level in enumerate(self.encoder_levels):
            # Apply Transformer blocks at CURRENT resolution (before downsampling)
            for transformer_block in level['transformer_blocks']:
                # Adjust padding mask for current resolution
                level_padding_mask = None
                if key_padding_mask is not None:
                    # Adjust padding mask for current resolution
                    level_padding_mask = key_padding_mask[:, ::2**i] if i > 0 else key_padding_mask

                x = transformer_block(x, level_padding_mask)

            # Store skip connection BEFORE downsampling (like UNet)
            skip_connections.append(x.clone())

            # Downsample for next level
            x = level['downsample'](x)

        # Apply bottleneck Transformer blocks
        for transformer_block in self.bottleneck:
            bottleneck_padding_mask = None
            if key_padding_mask is not None:
                bottleneck_padding_mask = key_padding_mask[:, ::2**self.num_layers]

            x = transformer_block(x, bottleneck_padding_mask)

        return x, skip_connections


class HierarchicalTransformerDecoder(nn.Module):
    """
    Hierarchical U-Net-style Transformer decoder.

    Progressive upsampling with skip connections from encoder.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        print(f"Building Hierarchical Transformer Decoder with {config.num_layers} levels")

        self.num_layers = config.num_layers
        self.encoder_dims = config.encoder_dims

        # Decoder levels with upsampling
        self.decoder_levels = nn.ModuleList()
        for i in reversed(range(self.num_layers)):
            in_dim = config.encoder_dims[i+1] if i+1 < len(config.encoder_dims) else config.encoder_dims[-1]
            out_dim = config.encoder_dims[i]

            # Upsampling layer
            upsample = Upsample1D(in_dim, out_dim)

            # Fusion layer (combine skip connection)
            # After concatenation: out_dim (from upsample) + out_dim (from skip) = 2*out_dim
            fusion = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU()
            )

            # Transformer blocks at this resolution
            transformer_blocks = nn.ModuleList([
                TransformerBlock(
                    d_model=out_dim,
                    nhead=config.nhead,
                    ff_dim=out_dim * 4,
                    dropout=config.dropout
                )
                for _ in range(config.num_transformer_blocks)
            ])

            self.decoder_levels.append(nn.ModuleDict({
                'upsample': upsample,
                'fusion': fusion,
                'transformer_blocks': transformer_blocks
            }))

        # Final output projection
        self.output_proj = nn.Linear(config.encoder_dims[0], 1)

    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical decoder.

        Args:
            x: Bottleneck features (batch, seq_len_small, d_model)
            skip_connections: List of skip connections from encoder (reversed order)
            key_padding_mask: Optional padding mask (batch, original_seq_len)

        Returns:
            Reconstructed output (batch, seq_len, 1)
        """
        # Reverse skip connections to match decoder order
        skip_connections = list(reversed(skip_connections))

        # Progressive upsampling with skip connections
        for i, level in enumerate(self.decoder_levels):
            # Upsample
            x = level['upsample'](x)

            # Get skip connection
            skip = skip_connections[i]

            # Handle size mismatches due to downsampling/upsampling
            if x.shape[1] != skip.shape[1]:
                # Interpolate to match sizes
                if x.shape[1] < skip.shape[1]:
                    x = F.interpolate(
                        x.transpose(1, 2),
                        size=skip.shape[1],
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
                else:
                    skip = F.interpolate(
                        skip.transpose(1, 2),
                        size=x.shape[1],
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)

            # Concatenate with skip connection
            x = torch.cat([x, skip], dim=-1)

            # Fusion
            x = level['fusion'](x)

            # Apply Transformer blocks at this resolution
            for transformer_block in level['transformer_blocks']:
                # Adjust padding mask for current resolution
                level_padding_mask = None
                if key_padding_mask is not None:
                    # Upsample padding mask to current resolution
                    current_scale = 2 ** (self.num_layers - i)
                    level_padding_mask = key_padding_mask[:, ::current_scale]
                    if level_padding_mask.shape[1] != x.shape[1]:
                        # Pad or truncate to match
                        if level_padding_mask.shape[1] < x.shape[1]:
                            pad_size = x.shape[1] - level_padding_mask.shape[1]
                            level_padding_mask = F.pad(level_padding_mask, (0, pad_size))
                        else:
                            level_padding_mask = level_padding_mask[:, :x.shape[1]]

                x = transformer_block(x, level_padding_mask)

        # Final output projection
        x = self.output_proj(x)

        return x


class TimeSeriesTransformer(BaseAutoencoder):
    """
    Hierarchical U-Net-style Transformer for time-series reconstruction.

    Features:
    - Multi-scale hierarchical processing (like UNet)
    - Skip connections from encoder to decoder
    - Pre-norm multi-head attention at each level
    - Time-aware positional encoding (optional)
    - Explicit mask channel
    - O(n²/4 + n²/16 + ...) complexity vs flat O(n²)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.config = config

        # Hierarchical encoder
        self.encoder = HierarchicalTransformerEncoder(config)

        # Hierarchical decoder
        self.decoder = HierarchicalTransformerDecoder(config)

        print(f"Hierarchical Transformer: {self.count_parameters():,} parameters")

    def encode(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode time series to latent representation.

        Args:
            x: Input (batch, seq_len, input_dim) where input_dim = [flux, mask]
            t: Timestamps (batch, seq_len) - optional for positional encoding
            key_padding_mask: Optional padding mask (batch, seq_len)

        Returns:
            encoded: Bottleneck representation (batch, seq_len_small, d_model)
            skip_connections: List of skip connections
        """
        return self.encoder(x, t, key_padding_mask)

    def decode(
        self,
        z: torch.Tensor,
        skip_connections: List[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent representation to flux values.

        Args:
            z: Latent representation (batch, seq_len_small, d_model)
            skip_connections: Skip connections from encoder
            key_padding_mask: Optional padding mask (batch, original_seq_len)

        Returns:
            Reconstructed flux (batch, seq_len, 1)
        """
        return self.decoder(z, skip_connections, key_padding_mask)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical transformer.

        Args:
            x: Input (batch, seq_len, input_dim) where input_dim = [flux, mask]
            t: Timestamps (batch, seq_len) - optional for positional encoding
            key_padding_mask: Optional padding mask (batch, seq_len)

        Returns:
            Dictionary with:
                - 'reconstructed': Reconstructed flux (batch, seq_len, 1)
                - 'encoded': Bottleneck representation
        """
        encoded, skip_connections = self.encode(x, t, key_padding_mask)
        reconstructed = self.decode(encoded, skip_connections, key_padding_mask)

        return {
            'reconstructed': reconstructed,
            'encoded': encoded
        }
