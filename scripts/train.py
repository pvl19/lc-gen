#!/usr/bin/env python
"""
Train models from command line using YAML configuration files.

Usage:
    python scripts/train.py --config configs/unet_ps.yaml
    python scripts/train.py --config configs/mlp_ps.yaml --epochs 200
    python scripts/train.py --config configs/transformer_lc.yaml --gpu 0
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lcgen.models import (
    PowerSpectrumUNetAutoencoder, UNetConfig,
    MLPAutoencoder, MLPConfig,
    TimeSeriesTransformer, TransformerConfig
)
from lcgen.data import (
    PowerSpectrumDataset, FluxDataset,
    normalize_ps, normalize_acf
)
from lcgen.training import (
    Trainer,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler
)
from lcgen.utils import load_config, save_config, set_seed


def create_model(config):
    """Create model from configuration"""
    model_type = config.model['model_type']

    if model_type == 'unet':
        model_config = UNetConfig(**config.model)
        model = PowerSpectrumUNetAutoencoder(model_config)
    elif model_type == 'mlp':
        model_config = MLPConfig(**config.model)
        model = MLPAutoencoder(model_config)
    elif model_type == 'transformer':
        model_config = TransformerConfig(**config.model)
        model = TimeSeriesTransformer(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Created {model_type} model with {model.count_parameters():,} parameters")
    return model


def load_data(config):
    """Load and prepare datasets"""
    import pickle
    import numpy as np

    data_path = Path(config.data.data_path)
    dataset_type = config.data.dataset_type

    print(f"Loading {dataset_type} data from {data_path}")

    if dataset_type in ['power_spectrum', 'acf']:
        # Load power spectrum or ACF data
        data_file = data_path / 'pk_star_lc_cf_norm.pickle'

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f)

        # Extract power spectra
        ps_data = np.array([data_dict[k]['power_1024_norm'] for k in data_dict.keys()])
        ps_data = ps_data.astype(np.float32)

        if dataset_type == 'acf':
            # Convert to ACF via inverse FFT
            data = np.fft.ifft(ps_data, axis=1).real
            # Normalize ACF
            normalized_data = np.array([
                normalize_acf(data[i], scale_factor=config.data.scale_factor)
                for i in range(len(data))
            ])
        else:
            # Normalize power spectra
            normalized_data = np.array([
                normalize_ps(ps_data[i], scale_factor=config.data.scale_factor)
                for i in range(len(ps_data))
            ])

        dataset = PowerSpectrumDataset(torch.from_numpy(normalized_data).float())

    elif dataset_type == 'flux':
        # Load light curve data
        data_file = data_path / 'star_dataset_subset.pt'

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        dataset = torch.load(data_file, weights_only=False)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"Loaded dataset with {len(dataset)} samples")
    return dataset


def create_data_loaders(dataset, config):
    """Split dataset and create data loaders"""
    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * config.data.test_split)
    train_val_size = total_size - test_size

    # Test split
    train_val_dataset, test_dataset = random_split(
        dataset,
        [train_val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # Train/val split
    val_size = int(train_val_size * config.data.val_split)
    train_size = train_val_size - val_size

    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    print(f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader


def create_callbacks(config, model):
    """Create training callbacks"""
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir=config.checkpoint_dir,
        monitor='val_loss',
        mode='min',
        save_best_only=config.training.save_best_only,
        save_every=config.training.save_every if not config.training.save_best_only else None,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if config.training.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=config.training.early_stopping_patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping_callback)

    return callbacks


def main():
    parser = argparse.ArgumentParser(description='Train LC-Gen models')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.gpu is not None:
        config.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'

    # Set random seed
    set_seed(config.seed)

    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, Path(config.output_dir) / 'config.yaml')
    print(f"Saved configuration to {config.output_dir}/config.yaml")

    # Create model
    model = create_model(config)

    # Load data
    dataset = load_data(config)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(dataset, config)

    # Create callbacks
    callbacks = create_callbacks(config, model)

    # Create learning rate scheduler
    scheduler = None
    if config.training.use_scheduler:
        if config.training.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate),
                mode='min',
                patience=config.training.scheduler_patience,
                factor=config.training.scheduler_factor
            )
        elif config.training.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate),
                T_max=config.training.epochs
            )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        scheduler=scheduler,
        callbacks=callbacks
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print(f"\nStarting training...")
    print(f"Experiment: {config.experiment_name} ({config.version})")
    print(f"Device: {config.get_device()}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print()

    history = trainer.train(train_loader, val_loader)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)

    # Save final results
    import numpy as np
    results = {
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses'],
        'learning_rates': history['learning_rates'],
        'test_metrics': test_metrics,
        'config': config.to_dict()
    }

    results_file = Path(config.output_dir) / f'{config.version}_results.pt'
    torch.save(results, results_file)
    print(f"\nSaved results to {results_file}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
