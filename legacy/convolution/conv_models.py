import torch
import torch.nn as nn
import torch.nn.functional as F
# from nn.Modules.conformer import ConformerEncoder, ConformerDecoder
# from nn.Modules.mhsa_pro import RotaryEmbedding, ContinuousRotaryEmbedding
# from nn.Modules.flash_mhsa import MHA as Flash_Mha
# from nn.Modules.mlp import Mlp as MLP
# from nn.simsiam import projection_MLP, SimSiam

import numbers
import torch.nn.init as init
from typing import Union, List, Optional, Tuple
from torch import Size, Tensor

def get_activation(args):
    
    if args.activation == 'silu':
        return nn.SiLU()
    elif args.activation == 'sine':
        return Sine(w0=args.sine_w0)
    elif args.activation == 'relu':
        return nn.ReLU()
    elif args.activation == 'gelu':
        return nn.GELU()
    elif args.activation == 'tanh':
        return nn.Tanh()
    elif args.activation == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()


class PowerSpectrumUNetAutoencoder(nn.Module):
    """
    UNet-based CNN Autoencoder for power spectrum reconstruction with skip connections.
    This architecture preserves fine-scale features through skip connections between
    encoder and decoder at corresponding resolution levels.
    """
    def __init__(self, args):
        super().__init__()
        
        self.encoder = PowerSpectrumUNetEncoder(args)
        self.decoder = PowerSpectrumUNetDecoder(args)
        self.latent_dim = args.encoder_dims[-1]
        
    def forward(self, x):
        # Encode the power spectrum and get skip connections
        bottleneck_features, skip_connections = self.encoder(x)
        # Decode with skip connections
        reconstructed = self.decoder(bottleneck_features, skip_connections)
        
        return {
            'reconstructed': reconstructed,
            'encoded': bottleneck_features,  # Now preserves spatial structure
            'skip_connections': skip_connections
        }
    
    def get_compact_latent(self, x):
        """
        Get a compact 1D latent representation (for analysis/visualization)
        by applying global pooling to the bottleneck features.
        """
        bottleneck_features, _ = self.encoder(x)
        # Apply global pooling only when explicitly requested
        compact_latent = torch.mean(bottleneck_features, dim=-1)  # [batch, channels]
        return compact_latent
    
    def encode_with_structure(self, x):
        """
        Get bottleneck features with preserved spatial structure
        """
        return self.encoder(x)


class PowerSpectrumUNetEncoder(nn.Module):
    """
    UNet-style CNN encoder for power spectra with skip connections.
    Stores intermediate feature maps at each downsampling level for skip connections.
    """
    def __init__(self, args):
        super().__init__()
        print(f"Using Power Spectrum UNet encoder with activation: {args.activation}")
        
        self.activation = get_activation(args)
        self.in_channels = getattr(args, 'in_channels', 1)
        self.num_layers = args.num_layers
        
        # Initial embedding layer
        self.embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=args.encoder_dims[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(args.encoder_dims[0]),
            self.activation,
        )
        
        # Downsampling blocks - each produces a skip connection
        self.down_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = args.encoder_dims[i] if i < len(args.encoder_dims) else args.encoder_dims[-1]
            out_dim = args.encoder_dims[i+1] if i+1 < len(args.encoder_dims) else args.encoder_dims[-1]
            
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
        bottleneck_dim = args.encoder_dims[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(bottleneck_dim),
            self.activation,
            nn.Conv1d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(bottleneck_dim),
            self.activation
        )
        
        # Calculate expected bottleneck spatial size for reference
        # After num_layers of 2x downsampling: input_length // (2^num_layers)
        expected_bottleneck_length = getattr(args, 'input_length', 1024) // (2 ** self.num_layers)
        self.expected_bottleneck_length = expected_bottleneck_length
        self.output_dim = args.encoder_dims[-1]
        
        print(f"Expected bottleneck spatial size: {expected_bottleneck_length}")
        
    def forward(self, x):
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
    def __init__(self, args):
        super().__init__()
        print(f"Using Power Spectrum UNet decoder with activation: {args.activation}")
        
        self.activation = get_activation(args)
        self.target_length = getattr(args, 'target_length', 1024)
        self.num_layers = args.num_layers
        
        # Reverse encoder dimensions for decoder
        decoder_dims = args.encoder_dims[::-1]
        
        # No initial expansion needed - we work directly with spatial bottleneck features
        
        # Upsampling blocks with skip connection integration
        self.up_blocks = nn.ModuleList()
        
        # Calculate actual skip connection dimensions based on encoder architecture
        # Skip connections come from encoder pre_conv outputs, which have these dimensions:
        skip_dims = []
        for i in range(self.num_layers):
            if i == 0:
                # First encoder block: input embedding (encoder_dims[0]) -> encoder_dims[1]
                skip_dims.append(args.encoder_dims[1] if len(args.encoder_dims) > 1 else args.encoder_dims[0])
            else:
                # Subsequent blocks: encoder_dims[i] -> encoder_dims[i+1]
                out_idx = min(i + 1, len(args.encoder_dims) - 1)
                skip_dims.append(args.encoder_dims[out_idx])
        
        for i in range(self.num_layers):
            in_dim = decoder_dims[i] if i < len(decoder_dims) else decoder_dims[-1]
            out_dim = decoder_dims[i+1] if i+1 < len(decoder_dims) else decoder_dims[-1]
            
            # Skip connection dimension - reverse order since skip_connections are [shallow->deep] but we process [deep->shallow]
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
            # Input channels = upsampled_channels + skip_connection_channels
            skip_integrate = nn.Sequential(
                nn.Conv1d(in_dim + skip_dim, out_dim, kernel_size=3, padding=1, bias=False),  # Concatenated channels
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
        final_in_dim = decoder_dims[-1] if len(decoder_dims) > 0 else args.encoder_dims[0]
        self.final_conv = nn.Sequential(
            nn.Conv1d(final_in_dim, final_in_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(final_in_dim // 2),
            self.activation,
            nn.Conv1d(final_in_dim // 2, 1, kernel_size=7, padding=3)
            # Removed ReLU since power spectra are normalized and can have any values
        )
        
        # Adaptive pooling to match target length exactly
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.target_length)
        
    def forward(self, bottleneck_features, skip_connections):
        # Start with bottleneck features (already has spatial structure)
        # Shape: [batch, channels, spatial_length]
        x = bottleneck_features
        
        # Progressive upsampling with skip connection integration
        # Note: skip_connections are in order from shallow to deep, so we reverse them
        skip_connections_reversed = skip_connections[::-1]
        
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
        
        # Ensure exact target length
        x = self.adaptive_pool(x)
        
        return x.squeeze(1)  # Return [batch, length]


class PowerSpectrumConfig:
    """
    Configuration class for PowerSpectrumAutoencoder and PowerSpectrumUNetAutoencoder
    
    Example usage:
    ```python
    # Create standard autoencoder configuration
    config = PowerSpectrumConfig(
        input_length=1024,      # Length of input power spectrum
        encoder_dims=[64, 128, 256, 512],  # Progressive encoding dimensions
        num_layers=4,           # Number of encoding layers
        activation='gelu'       # Activation function
    )
    
    # Create UNet autoencoder
    unet_model = PowerSpectrumUNetAutoencoder(config)
    
    # Forward pass
    power_spectrum = torch.randn(32, 1024)  # Batch of power spectra
    output = unet_model(power_spectrum)
    reconstructed = output['reconstructed']  # Reconstructed power spectrum
    encoded = output['encoded']              # Latent representation
    skip_connections = output['skip_connections']  # Skip connections for analysis
    
    # Training with MSE loss
    criterion = nn.MSELoss()
    loss = criterion(reconstructed, power_spectrum)
    ```
    """
    def __init__(self, 
                 input_length=1024,
                 target_length=None,
                 encoder_dims=[64, 128, 256, 512],
                 num_layers=4,
                 activation='gelu',
                 in_channels=1,
                 use_unet=False):
        self.input_length = input_length
        self.target_length = target_length or input_length
        self.encoder_dims = encoder_dims
        self.num_layers = min(num_layers, len(encoder_dims))
        self.activation = activation
        self.in_channels = in_channels
        self.use_unet = use_unet

