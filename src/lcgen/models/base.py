"""Base classes for all model architectures"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """
    Base configuration class for all models.

    Subclasses should extend this with model-specific parameters.
    """
    # Model identification
    model_type: str = "base"
    model_name: str = "base_model"

    # Device
    device: str = "auto"  # 'cuda', 'cpu', or 'auto'

    def __post_init__(self):
        """Post-initialization hook for subclasses"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary"""
        return cls(**config_dict)

    def get_device(self) -> torch.device:
        """Get the torch device based on config"""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class BaseAutoencoder(nn.Module, ABC):
    """
    Abstract base class for all autoencoder models.

    All model implementations should inherit from this class and implement
    the abstract methods.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = config.get_device()

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor

        Returns:
            Latent representation
        """
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent representation

        Returns:
            Reconstructed output
        """
        pass

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing:
                - 'reconstructed': Reconstructed output
                - 'encoded': Latent representation
                - Additional model-specific outputs
        """
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)

        return {
            'reconstructed': reconstructed,
            'encoded': encoded
        }

    def get_latent_dim(self) -> int:
        """Return the dimensionality of the latent space"""
        return self.config.latent_dim

    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio (input dim / latent dim).

        For models where this doesn't make sense, return 1.0
        """
        if hasattr(self.config, 'input_dim'):
            return self.config.input_dim / self.config.latent_dim
        return 1.0

    def save_checkpoint(self, path: str, epoch: int = 0, optimizer_state: Optional[Dict] = None,
                       loss: float = 0.0, metadata: Optional[Dict] = None):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict (optional)
            loss: Current loss value
            metadata: Additional metadata to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'loss': loss,
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        if metadata is not None:
            checkpoint['metadata'] = metadata

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, config_class: type, device: Optional[str] = None):
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            config_class: Configuration class for this model type
            device: Device to load model on (optional)

        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=device or 'cpu')

        # Reconstruct config
        config = config_class.from_dict(checkpoint['config'])

        # Create model instance
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        if device is not None:
            model = model.to(device)

        return model, checkpoint

    def to(self, device):
        """Override to() to also update internal device tracking"""
        self.device = torch.device(device) if isinstance(device, str) else device
        return super().to(device)

    def __repr__(self) -> str:
        """String representation with key model info"""
        param_count = self.count_parameters()
        return (f"{self.__class__.__name__}(\n"
                f"  config={self.config.model_name},\n"
                f"  parameters={param_count:,},\n"
                f"  latent_dim={self.get_latent_dim()},\n"
                f"  device={self.device}\n"
                f")")


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for encoder modules"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        pass

    def get_output_dim(self) -> int:
        """Return output dimension of encoder"""
        raise NotImplementedError


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for decoder modules"""

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output"""
        pass

    def get_input_dim(self) -> int:
        """Return expected input dimension"""
        raise NotImplementedError
