"""Unified training infrastructure for all model types"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
from tqdm import tqdm

from .losses import get_loss_fn
from .callbacks import CallbackList, Callback
from ..utils.config import Config
from ..utils.logging import TrainingLogger


class Trainer:
    """
    Unified trainer for all autoencoder models.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None
    ):
        """
        Args:
            model: Model to train
            config: Training configuration
            optimizer: Optimizer (if None, creates AdamW)
            scheduler: Learning rate scheduler (optional)
            callbacks: List of callbacks (optional)
        """
        self.model = model
        self.config = config
        self.device = config.get_device()

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer

        # Setup scheduler
        self.scheduler = scheduler

        # Setup callbacks
        self.callbacks = CallbackList(callbacks or [])

        # Setup loss function
        self.loss_fn = get_loss_fn(config.training.loss_fn)

        # Setup logger
        self.logger = TrainingLogger(
            log_dir=config.log_dir,
            experiment_name=config.experiment_name
        )

        # Training state
        self.current_epoch = 0
        self.stop_training = False  # For early stopping
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config"""
        optimizer_name = self.config.training.optimizer.lower()

        if optimizer_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs (overrides config if provided)

        Returns:
            Dictionary with training history (train_losses, val_losses, learning_rates)
        """
        epochs = epochs or self.config.training.epochs

        print(f"Training for {epochs} epochs on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")

        # Callbacks
        self.callbacks.on_train_begin(self)

        start_time = time.time()

        for epoch in range(epochs):
            if self.stop_training:
                break

            self.current_epoch = epoch

            # Epoch start
            self.callbacks.on_epoch_begin(epoch, self)

            # Train one epoch
            train_loss = self._train_epoch(train_loader, epoch)

            # Validate
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
            else:
                val_loss = train_loss  # Use train loss if no validation set

            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Store learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            }

            self.logger.log_metrics(metrics, epoch)

            # Epoch end
            self.callbacks.on_epoch_end(epoch, metrics, self)

            # Print progress
            if (epoch + 1) % self.config.training.log_every == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss = {train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}, "
                      f"LR = {current_lr:.2e}")

        # Training end
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        print(f"Best validation loss: {min(self.val_losses):.6f}")

        self.callbacks.on_train_end(self)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        # Progress bar (optional)
        if self.config.training.log_every == 1:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        else:
            iterator = train_loader

        for batch_idx, batch in enumerate(iterator):
            # Callbacks
            self.callbacks.on_batch_begin(batch_idx, self)

            # Forward pass
            loss = self._train_step(batch)

            # Store loss
            epoch_loss += loss.item()

            # Callbacks
            self.callbacks.on_batch_end(batch_idx, loss.item(), self)

            # Update progress bar
            if isinstance(iterator, tqdm):
                iterator.set_postfix({'loss': f'{loss.item():.6f}'})

        return epoch_loss / num_batches

    def _train_step(self, batch) -> torch.Tensor:
        """Single training step"""
        # Move batch to device
        if isinstance(batch, (tuple, list)):
            batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            x = batch[0]
        else:
            x = batch.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(x)
        reconstructed = output['reconstructed']

        # Compute loss
        loss = self.loss_fn(reconstructed, x)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional)
        if self.config.training.clip_gradients:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )

        # Optimizer step
        self.optimizer.step()

        return loss

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                    x = batch[0]
                else:
                    x = batch.to(self.device)

                # Forward pass
                output = self.model(x)
                reconstructed = output['reconstructed']

                # Compute loss
                loss = self.loss_fn(reconstructed, x)
                epoch_loss += loss.item()

        return epoch_loss / num_batches

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'is_best': is_best
        }

        if metrics is not None:
            checkpoint['metrics'] = metrics

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint and resume training.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler if exists
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of evaluation metrics
        """
        from ..evaluation import evaluate_reconstruction

        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                    x = batch[0]
                else:
                    x = batch.to(self.device)

                # Forward pass
                output = self.model(x)
                reconstructed = output['reconstructed']

                all_predictions.append(reconstructed.cpu())
                all_targets.append(x.cpu())

        # Concatenate all batches
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        metrics = evaluate_reconstruction(predictions, targets)

        print("\nEvaluation Results:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")

        return metrics
