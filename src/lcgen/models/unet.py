"""UNet-based CNN Autoencoder for Power Spectra and ACF"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .base import BaseAutoencoder, ModelConfig


@dataclass
class UNetConfig(ModelConfig):
    """Configuration for UNet autoencoder"""
    model_type: str = "unet"
    model_name: str = "unet_autoencoder"

    # Architecture
    input_length: int = 1024
    target_length: int = 1024
    in_channels: int = 1
    encoder_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_layers: int = 4
    activation: str = "gelu"

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


def get_activation(activation_name: str, **kwargs):
    """Get activation function by name"""
    activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'tanh': nn.Tanh,
        'leakyrelu': nn.LeakyReLU,
    }

    if activation_name.lower() not in activations:
        raise ValueError(f"Unknown activation: {activation_name}. Choose from {list(activations.keys())}")

    return activations[activation_name.lower()](**kwargs)


class PowerSpectrumUNetEncoder(nn.Module):
    """
    UNet-style CNN encoder for power spectra with skip connections.

    Progressive downsampling with skip connection storage at each level.
    """

    def __init__(self, config: UNetConfig):
        super().__init__()
        print(f"Using Power Spectrum UNet encoder with activation: {config.activation}")

        self.activation = get_activation(config.activation)
        self.in_channels = config.in_channels
        self.num_layers = config.num_layers

        # Initial embedding layer
        self.embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=config.encoder_dims[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(config.encoder_dims[0]),
            self.activation,
        )

        # Downsampling blocks - each produces a skip connection
        self.down_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = config.encoder_dims[i] if i < len(config.encoder_dims) else config.encoder_dims[-1]
            out_dim = config.encoder_dims[i+1] if i+1 < len(config.encoder_dims) else config.encoder_dims[-1]

            # Pre-downsampling convolution (skip connection source)
            pre_conv = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_dim),
                self.activation,
                nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_dim),
                self.activation
            )

            # Downsampling
            downsample = nn.MaxPool1d(kernel_size=2, stride=2)

            self.down_blocks.append(nn.ModuleDict({
                'pre_conv': pre_conv,
                'downsample': downsample
            }))

        # Bottleneck layer (deepest part of UNet)
        bottleneck_dim = config.encoder_dims[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(bottleneck_dim),
            self.activation,
            nn.Conv1d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(bottleneck_dim),
            self.activation
        )

        # Calculate expected bottleneck spatial size for reference
        expected_bottleneck_length = config.input_length // (2 ** self.num_layers)
        self.expected_bottleneck_length = expected_bottleneck_length
        self.output_dim = config.encoder_dims[-1]

        print(f"Expected bottleneck spatial size: {expected_bottleneck_length}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor

        Returns:
            bottleneck_features: Encoded representation with spatial structure
            skip_connections: List of skip connection tensors
        """
        # Handle different input dimensions
        if len(x.shape) == 2:  # [batch, length] -> [batch, 1, length]
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:  # [batch, length, 1] -> [batch, 1, length]
            x = x.permute(0, 2, 1)

        x = x.float()

        # Initial embedding
        x = self.embedding(x)

        # Store skip connections at each level
        skip_connections = []

        # Progressive downsampling with skip connection storage
        for i, block in enumerate(self.down_blocks):
            # Apply pre-downsampling convolutions
            x = block['pre_conv'](x)

            # Store skip connection BEFORE downsampling
            skip_connections.append(x.clone())

            # Downsample for next level
            x = block['downsample'](x)

        # Apply bottleneck
        bottleneck_features = self.bottleneck(x)

        # Return bottleneck features with spatial structure preserved
        # Shape: [batch, channels, spatial_length]
        return bottleneck_features, skip_connections


class PowerSpectrumUNetDecoder(nn.Module):
    """
    UNet-style CNN decoder for power spectra with skip connections.

    Uses skip connections from encoder to preserve fine-scale features.
    """

    def __init__(self, config: UNetConfig):
        super().__init__()
        print(f"Using Power Spectrum UNet decoder with activation: {config.activation}")

        self.activation = get_activation(config.activation)
        self.target_length = config.target_length
        self.num_layers = config.num_layers

        # Reverse encoder dimensions for decoder
        decoder_dims = config.encoder_dims[::-1]

        # Upsampling blocks with skip connection integration
        self.up_blocks = nn.ModuleList()

        # Calculate actual skip connection dimensions
        skip_dims = []
        for i in range(self.num_layers):
            if i == 0:
                skip_dims.append(config.encoder_dims[1] if len(config.encoder_dims) > 1 else config.encoder_dims[0])
            else:
                out_idx = min(i + 1, len(config.encoder_dims) - 1)
                skip_dims.append(config.encoder_dims[out_idx])

        for i in range(self.num_layers):
            in_dim = decoder_dims[i] if i < len(decoder_dims) else decoder_dims[-1]
            out_dim = decoder_dims[i+1] if i+1 < len(decoder_dims) else decoder_dims[-1]

            # Skip connection dimension - reverse order
            skip_idx = self.num_layers - 1 - i
            skip_dim = skip_dims[skip_idx] if skip_idx < len(skip_dims) else skip_dims[-1]

            # Upsampling layer
            upsample = nn.ConvTranspose1d(
                in_channels=in_dim,
                out_channels=in_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )

            # Skip connection integration + convolution
            skip_integrate = nn.Sequential(
                nn.Conv1d(in_dim + skip_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_dim),
                self.activation,
                nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_dim),
                self.activation
            )

            self.up_blocks.append(nn.ModuleDict({
                'upsample': upsample,
                'skip_integrate': skip_integrate
            }))

        # Final reconstruction layer
        final_in_dim = decoder_dims[-1] if len(decoder_dims) > 0 else config.encoder_dims[0]
        self.final_conv = nn.Sequential(
            nn.Conv1d(final_in_dim, final_in_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(final_in_dim // 2),
            self.activation,
            nn.Conv1d(final_in_dim // 2, 1, kernel_size=7, padding=3)
        )

        # Adaptive pooling to match target length exactly
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.target_length)

    def forward(self, bottleneck_features: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            bottleneck_features: Encoded representation
            skip_connections: Skip connections from encoder

        Returns:
            Reconstructed output of shape [batch, length]
        """
        x = bottleneck_features

        # Reverse skip connections (shallow to deep -> deep to shallow)
        skip_connections_reversed = skip_connections[::-1]

        # Progressive upsampling with skip connection integration
        for i, (block, skip) in enumerate(zip(self.up_blocks, skip_connections_reversed)):
            # Upsample current feature map
            x = block['upsample'](x)

            # Interpolate skip connection to match current spatial size
            if x.shape[-1] != skip.shape[-1]:
                skip = F.interpolate(skip, size=x.shape[-1], mode='linear', align_corners=False)

            # Concatenate upsampled features with skip connection
            x = torch.cat([x, skip], dim=1)

            # Integrate skip connection with convolutions
            x = block['skip_integrate'](x)

        # Final reconstruction
        x = self.final_conv(x)

        # Ensure exact target length
        x = self.adaptive_pool(x)

        return x.squeeze(1)  # Return [batch, length]


class PowerSpectrumUNetAutoencoder(BaseAutoencoder):
    """
    UNet-based CNN Autoencoder for power spectrum reconstruction with skip connections.

    This architecture preserves fine-scale features through skip connections between
    encoder and decoder at corresponding resolution levels.
    """

    def __init__(self, config: UNetConfig):
        super().__init__(config)

        self.encoder_module = PowerSpectrumUNetEncoder(config)
        self.decoder_module = PowerSpectrumUNetDecoder(config)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation (bottleneck features).

        Args:
            x: Input power spectrum/ACF

        Returns:
            Bottleneck features with spatial structure preserved
        """
        bottleneck_features, _ = self.encoder_module(x)
        return bottleneck_features

    def decode(self, z: torch.Tensor, skip_connections: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Note: This requires skip_connections for proper UNet decoding.
        Use forward() for end-to-end reconstruction.

        Args:
            z: Latent representation (bottleneck features)
            skip_connections: Skip connections from encoder (required)

        Returns:
            Reconstructed power spectrum/ACF
        """
        if skip_connections is None:
            raise ValueError("UNet decoder requires skip_connections. Use forward() instead.")
        return self.decoder_module(z, skip_connections)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through autoencoder.

        Args:
            x: Input power spectrum/ACF

        Returns:
            Dictionary containing:
                - 'reconstructed': Reconstructed output
                - 'encoded': Bottleneck features
                - 'skip_connections': Skip connections for analysis
        """
        # Encode with skip connections
        bottleneck_features, skip_connections = self.encoder_module(x)

        # Decode with skip connections
        reconstructed = self.decoder_module(bottleneck_features, skip_connections)

        return {
            'reconstructed': reconstructed,
            'encoded': bottleneck_features,
            'skip_connections': skip_connections
        }

    def get_compact_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get a compact 1D latent representation by global pooling.

        Useful for analysis/visualization when spatial structure is not needed.

        Args:
            x: Input power spectrum/ACF

        Returns:
            Compact latent vector of shape [batch, channels]
        """
        bottleneck_features, _ = self.encoder_module(x)
        # Apply global pooling
        compact_latent = torch.mean(bottleneck_features, dim=-1)  # [batch, channels]
        return compact_latent
