"""Lightweight 1D CNN encoder for power spectra and ACFs.

Much faster and simpler than UNet for encoding frequency-domain features.
"""
import torch
import torch.nn as nn


class LightweightConv1DEncoder(nn.Module):
    """Simple 1D CNN encoder for frequency-domain data.

    Uses strided convolutions for efficient downsampling without complex UNet architecture.
    Much faster than UNet while still capturing frequency-domain patterns.
    """
    def __init__(self, in_channels=1, hidden_channels=16, num_layers=3, activation='gelu'):
        super().__init__()

        if activation == 'gelu':
            act = nn.GELU()
        elif activation == 'relu':
            act = nn.ReLU()
        else:
            act = nn.SiLU()

        layers = []
        current_channels = in_channels

        # Progressive downsampling with strided convolutions
        for i in range(num_layers):
            out_channels = hidden_channels * (2 ** i)  # Double channels each layer

            layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size=8, stride=4, padding=2),
                nn.BatchNorm1d(out_channels),
                act,
            ])
            current_channels = out_channels

        self.encoder = nn.Sequential(*layers)
        self.output_channels = current_channels

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, length) input tensor

        Returns:
            encoded: (B, output_channels, downsampled_length) encoded features
        """
        # Handle 2D input (B, length) -> (B, 1, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        encoded = self.encoder(x)
        return encoded
