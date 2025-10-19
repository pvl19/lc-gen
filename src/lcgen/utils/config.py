"""Configuration management utilities"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import torch


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    data_path: str = "./data"
    dataset_type: str = "power_spectrum"  # 'power_spectrum', 'acf', 'flux', 'multimodal'
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    val_split: float = 0.2
    test_split: float = 0.1

    # Preprocessing
    normalize: bool = True
    normalization_method: str = "arcsinh"  # 'arcsinh', 'zscore', 'minmax'
    scale_factor: float = 50.0  # For arcsinh normalization

    # Masking (for training)
    use_masking: bool = False
    mask_ratio: float = 0.5
    mask_block_size: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Configuration for training"""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # 'adam', 'adamw', 'sgd'

    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "plateau"  # 'plateau', 'cosine', 'step'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Loss function
    loss_fn: str = "mse"  # 'mse', 'mae', 'chi_squared', 'masked_mse'

    # Checkpointing
    save_every: int = 10
    save_best_only: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 20

    # Logging
    log_every: int = 1
    plot_examples: bool = True
    plot_every: int = 10

    # Mixed precision training
    mixed_precision: bool = False

    # Gradient clipping
    clip_gradients: bool = False
    max_grad_norm: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)


@dataclass
class Config:
    """
    Master configuration class combining all sub-configs.

    This is the main config object used throughout the codebase.
    """
    # Sub-configurations
    model: Dict[str, Any] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Paths
    output_dir: str = "./experiments/results"
    checkpoint_dir: str = "./experiments/results/checkpoints"
    log_dir: str = "./experiments/results/logs"

    # Experiment tracking
    experiment_name: str = "experiment"
    version: str = "v01"
    notes: str = ""

    # Device
    device: str = "auto"  # 'cuda', 'cpu', or 'auto'
    seed: int = 42

    def __post_init__(self):
        """Convert dict configs to dataclass objects"""
        if isinstance(self.data, dict):
            self.data = DataConfig.from_dict(self.data)
        if isinstance(self.training, dict):
            self.training = TrainingConfig.from_dict(self.training)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary"""
        return {
            'model': self.model,
            'data': self.data.to_dict(),
            'training': self.training.to_dict(),
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'experiment_name': self.experiment_name,
            'version': self.version,
            'notes': self.notes,
            'device': self.device,
            'seed': self.seed,
        }

    def get_device(self) -> torch.device:
        """Get the torch device"""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Handle None case (empty YAML)
    if config_dict is None:
        config_dict = {}

    return Config(**config_dict)


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save
        save_path: Path where to save the YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """
    Merge base configuration with override dictionary.

    Useful for command-line argument overrides.

    Args:
        base_config: Base configuration
        override_dict: Dictionary with override values

    Returns:
        Merged configuration
    """
    base_dict = base_config.to_dict()

    # Deep merge
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    merged_dict = deep_update(base_dict, override_dict)
    return Config(**merged_dict)


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
