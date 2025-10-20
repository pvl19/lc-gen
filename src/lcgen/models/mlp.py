"""MLP-based Autoencoder for Power Spectra and ACF"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional

from .base import BaseAutoencoder, ModelConfig


@dataclass
class MLPConfig(ModelConfig):
    """Configuration for MLP autoencoder"""
    model_type: str = "mlp"
    model_name: str = "mlp_autoencoder"

    # Architecture
    input_dim: int = 2048  # 1024 data + 1024 binary mask
    output_dim: int = 1024  # Reconstruct only the data, not the mask
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    latent_dim: int = 64
    decoder_hidden_dims: Optional[List[int]] = None  # If None, use reverse of encoder
    dropout: float = 0.0
    activation: str = "gelu"


class MLPBlock(nn.Module):
    """Basic MLP block with layer normalization and dropout"""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'leakyrelu': nn.LeakyReLU(),
        }
        self.activation = activations.get(activation.lower(), nn.ReLU())

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MLPEncoder(nn.Module):
    """MLP-based encoder for power spectrum data"""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int,
                 dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build encoder layers
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(MLPBlock(current_dim, hidden_dim, dropout, activation))
            current_dim = hidden_dim

        # Final layer to latent space (no activation/dropout on final layer)
        layers.append(nn.Linear(current_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class MLPDecoder(nn.Module):
    """MLP-based decoder for power spectrum reconstruction"""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Build decoder layers (reverse of encoder)
        layers = []
        current_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(MLPBlock(current_dim, hidden_dim, dropout, activation))
            current_dim = hidden_dim

        # Final reconstruction layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class MLPAutoencoder(BaseAutoencoder):
    """
    Complete MLP-based autoencoder for power spectrum data.

    Uses fully-connected layers with layer normalization and dropout.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        # Use symmetric architecture if decoder dims not specified
        decoder_hidden_dims = config.decoder_hidden_dims
        if decoder_hidden_dims is None:
            decoder_hidden_dims = config.encoder_hidden_dims[::-1]  # Reverse encoder dims

        self.encoder_module = MLPEncoder(
            config.input_dim,
            config.encoder_hidden_dims,
            config.latent_dim,
            config.dropout,
            config.activation
        )

        self.decoder_module = MLPDecoder(
            config.latent_dim,
            decoder_hidden_dims,
            config.output_dim,
            config.dropout,
            config.activation
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with Xavier uniform"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.

        Args:
            x: Input power spectrum/ACF

        Returns:
            Latent representation
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)

        return self.encoder_module(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent representation

        Returns:
            Reconstructed power spectrum/ACF
        """
        return self.decoder_module(z)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through autoencoder.

        Args:
            x: Input power spectrum/ACF

        Returns:
            Dictionary containing:
                - 'reconstructed': Reconstructed output
                - 'encoded': Latent representation
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)

        encoded = self.encode(x)
        reconstructed = self.decode(encoded)

        return {
            'reconstructed': reconstructed,
            'encoded': encoded
        }
