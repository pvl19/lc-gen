"""Transformer models for time-series light curve reconstruction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional

from .base import BaseAutoencoder, ModelConfig


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for Transformer models"""
    model_type: str = "transformer"
    model_name: str = "transformer_lc"

    # Architecture
    input_dim: int = 2  # flux + flux_err
    embed_dim: int = 128
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.1

    # Positional encoding
    min_period: float = 0.00278  # Minimum period in days
    max_period: float = 1640.0   # Maximum period in days

    # Masking
    use_learnable_mask: bool = False  # Use learnable mask tokens

    @property
    def latent_dim(self) -> int:
        """Latent dimension is d_model"""
        return self.d_model


class TimeChannelPositionalEncoding(nn.Module):
    """
    Positional encoding based on actual time values (not indices).

    Uses log-spaced frequencies to encode time information across
    multiple scales.
    """

    def __init__(self, embed_dim: int, min_period: float, max_period: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.min_period = min_period
        self.max_period = max_period

        # Precompute angular frequencies (log-spaced)
        num_freqs = embed_dim // 2
        periods = torch.logspace(np.log10(min_period), np.log10(max_period), num_freqs)
        self.register_buffer("div_term", 2 * np.pi / periods)  # shape: (num_freqs,)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings (batch, seq_len, embed_dim)
            t: Time values (batch, seq_len, 1) - real timestamps

        Returns:
            x + positional encoding
        """
        batch_size, seq_len, _ = x.shape

        # Normalize time to [0,1] for stability
        t_min, t_max = t.min(), t.max()
        t_norm = (t - t_min) / (t_max - t_min + 1e-8)

        # Compute angles: (batch, seq_len, num_freqs)
        angles = t_norm * self.div_term.unsqueeze(0).unsqueeze(0)

        # Sin/cos encoding: (batch, seq_len, embed_dim)
        pe = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)

        return x + pe


class SingleHeadAttention(nn.Module):
    """Simple single-head attention mechanism"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-head self-attention.

        Args:
            x: Input (batch, seq_len, embed_dim)

        Returns:
            output: Attention output
            attn_weights: Attention weights
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum
        out = torch.bmm(attn_weights, V)
        return out, attn_weights


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block"""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = SingleHeadAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual connections"""
        # Attention + Residual
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feedforward + Residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks"""

    def __init__(self, embed_dim: int, ff_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Forward through all layers, collecting attention maps"""
        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_maps.append(attn_weights)
        return x, attn_maps


class TimeSeriesTransformer(BaseAutoencoder):
    """
    Basic transformer for light curve time series.

    Uses time-aware positional encoding based on actual timestamps.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.input_proj = nn.Linear(config.input_dim, config.embed_dim)
        self.pos_encoding = TimeChannelPositionalEncoding(
            config.embed_dim,
            config.min_period,
            config.max_period
        )
        self.encoder_module = TransformerEncoder(
            config.embed_dim,
            config.ff_dim,
            config.num_layers,
            config.dropout
        )
        self.output_proj = nn.Linear(config.embed_dim, 1)  # Output flux only

    def encode(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode time series to latent representation.

        Args:
            x: Input (batch, seq_len, channels) or (batch, seq_len, 2)
            t: Timestamps (batch, seq_len, 1)

        Returns:
            Encoded representation
        """
        if t is None:
            # If no timestamps provided, use indices
            seq_len = x.shape[1]
            t = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
            t = t.expand(x.shape[0], -1, -1).float()

        x = self.input_proj(x)
        x = self.pos_encoding(x, t)
        encoded, _ = self.encoder_module(x)
        return encoded

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to flux values.

        Args:
            z: Latent representation

        Returns:
            Reconstructed flux
        """
        return self.output_proj(z)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass through transformer.

        Args:
            x: Input (batch, seq_len, channels)
            t: Timestamps (batch, seq_len, 1)

        Returns:
            Dictionary with reconstructed flux and encoded representation
        """
        encoded = self.encode(x, t)
        reconstructed = self.decode(encoded)

        return {
            'reconstructed': reconstructed,
            'encoded': encoded
        }


class MaskedTimeSeriesTransformer(BaseAutoencoder):
    """
    Transformer with learnable mask tokens for masked reconstruction.

    Useful for training with masked autoencoding objectives.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.d_model)

        # Learnable mask token - one per feature dimension
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output projection back to original space
        self.output_projection = nn.Linear(config.d_model, 1)

    def create_positional_encoding(self, timestamps: torch.Tensor, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding based on actual timestamps.

        Args:
            timestamps: (batch, seq_len) - actual time values
            d_model: Dimension of model

        Returns:
            Positional encoding (batch, seq_len, d_model)
        """
        batch_size, seq_len = timestamps.shape
        pe = torch.zeros(batch_size, seq_len, d_model, device=timestamps.device)

        # Create sinusoidal features
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=timestamps.device) *
            -(np.log(10000.0) / d_model)
        )

        # Expand timestamps for broadcasting
        pos = timestamps.unsqueeze(-1)  # [batch, seq_len, 1]

        # Apply sinusoidal transformation
        pe[..., 0::2] = torch.sin(pos * div_term)
        pe[..., 1::2] = torch.cos(pos * div_term)

        return pe

    def encode(self, x: torch.Tensor, timestamps: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode time series with optional masking.

        Args:
            x: Input (batch, seq_len, input_dim)
            timestamps: Timestamps (batch, seq_len)
            mask: Boolean mask (batch, seq_len), True for masked positions

        Returns:
            Encoded representation
        """
        batch_size, seq_len, _ = x.shape

        # Project input to model dimension
        x_projected = self.input_projection(x)

        # Add positional encoding
        pos_encoding = self.create_positional_encoding(timestamps, self.input_projection.out_features)
        x_with_pos = x_projected + pos_encoding

        # Apply mask tokens if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x_with_pos)
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # Keep positional encoding but replace values with mask token
            x_with_pos = torch.where(
                mask_expanded,
                mask_tokens + pos_encoding,  # mask token + positional info
                x_with_pos
            )

        # Pass through transformer
        transformer_out = self.transformer(x_with_pos)

        return transformer_out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to flux"""
        return self.output_projection(z)

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional masking.

        Args:
            x: Input (batch, seq_len, input_dim)
            timestamps: Timestamps (batch, seq_len)
            mask: Boolean mask (batch, seq_len), True for masked positions

        Returns:
            output: Reconstructed flux
            mask: Mask used (for loss calculation)
        """
        encoded = self.encode(x, timestamps, mask)
        output = self.decode(encoded)

        return output, mask
