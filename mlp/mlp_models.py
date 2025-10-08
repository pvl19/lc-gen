import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLPBlock(nn.Module):
    """Basic MLP block with layer normalization and dropout"""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
    
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
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)


class MLPAutoencoder(nn.Module):
    """Complete MLP-based autoencoder for power spectrum data"""
    def __init__(self, input_dim: int = 1024, 
                 encoder_hidden_dims: List[int] = [512, 256, 128], 
                 latent_dim: int = 64,
                 decoder_hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Use symmetric architecture if decoder dims not specified
        if decoder_hidden_dims is None:
            decoder_hidden_dims = encoder_hidden_dims[::-1]  # Reverse encoder dims
        
        self.encoder = MLPEncoder(input_dim, encoder_hidden_dims, latent_dim, dropout, activation)
        self.decoder = MLPDecoder(latent_dim, decoder_hidden_dims, input_dim, dropout, activation)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with Xavier uniform"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through autoencoder"""
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
    
    def get_compression_ratio(self):
        """Calculate compression ratio"""
        return self.input_dim / self.latent_dim
    
    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPConfig:
    """Configuration class for MLP Autoencoder"""
    def __init__(self,
                 input_dim: int = 1024,
                 encoder_hidden_dims: List[int] = [512, 256, 128],
                 latent_dim: int = 64,
                 decoder_hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        self.input_dim = input_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.latent_dim = latent_dim
        self.decoder_hidden_dims = decoder_hidden_dims
        self.dropout = dropout
        self.activation = activation
