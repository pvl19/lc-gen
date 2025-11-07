"""Training callbacks for model checkpointing, early stopping, and logging"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
import numpy as np


class Callback:
    """Base class for training callbacks"""

    def on_train_begin(self, trainer):
        """Called at the beginning of training"""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training"""
        pass

    def on_epoch_begin(self, epoch: int, trainer):
        """Called at the beginning of each epoch"""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Called at the end of each epoch"""
        pass

    def on_batch_begin(self, batch_idx: int, trainer):
        """Called at the beginning of each batch"""
        pass

    def on_batch_end(self, batch_idx: int, loss: float, trainer):
        """Called at the end of each batch"""
        pass


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints during training.

    Saves best model based on monitored metric and/or periodic checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_every: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor ('val_loss', 'train_loss', etc.)
            mode: 'min' or 'max' - whether lower/higher is better
            save_best_only: Only save when metric improves
            save_every: Save checkpoint every N epochs (None = only best)
            verbose: Print when saving checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_every = save_every
        self.verbose = verbose

        # Track best metric
        if mode == 'min':
            self.best_metric = float('inf')
            self.is_better = lambda new, old: new < old
        else:
            self.best_metric = float('-inf')
            self.is_better = lambda new, old: new > old

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Save checkpoint if conditions are met"""
        current_metric = metrics.get(self.monitor)

        if current_metric is None:
            if self.verbose:
                print(f"Warning: Monitored metric '{self.monitor}' not found in metrics")
            return

        # Check if this is the best model
        is_best = self.is_better(current_metric, self.best_metric)

        if is_best:
            self.best_metric = current_metric

            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved to {current_metric:.6f}, saving model")

            # Save best model
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
            trainer.save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                metrics=metrics,
                is_best=True
            )

        # Periodic saving
        if self.save_every is not None and (epoch + 1) % self.save_every == 0:
            if not self.save_best_only or not is_best:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                if self.verbose:
                    print(f"Epoch {epoch}: Saving periodic checkpoint")
                trainer.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    metrics=metrics,
                    is_best=False
                )


class EarlyStopping(Callback):
    """
    Stop training when monitored metric stops improving.
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            verbose: Print when stopping
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        # Track best metric and patience counter
        if mode == 'min':
            self.best_metric = float('inf')
            self.is_better = lambda new, old: (old - new) > self.min_delta
        else:
            self.best_metric = float('-inf')
            self.is_better = lambda new, old: (new - old) > self.min_delta

        self.epochs_no_improve = 0
        self.stopped_epoch = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Check if training should stop"""
        current_metric = metrics.get(self.monitor)

        if current_metric is None:
            return

        if self.is_better(current_metric, self.best_metric):
            # Improvement
            self.best_metric = current_metric
            self.epochs_no_improve = 0
        else:
            # No improvement
            self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True

                if self.verbose:
                    print(f"\nEarly stopping triggered after epoch {epoch}")
                    print(f"No improvement in {self.monitor} for {self.patience} epochs")
                    print(f"Best {self.monitor}: {self.best_metric:.6f}")

    def on_train_end(self, trainer):
        """Print summary on training end"""
        if self.verbose and self.stopped_epoch is not None:
            print(f"Training stopped early at epoch {self.stopped_epoch}")


class LearningRateScheduler(Callback):
    """
    Callback to adjust learning rate during training.
    """

    def __init__(
        self,
        scheduler,
        monitor: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Args:
            scheduler: PyTorch learning rate scheduler instance
            monitor: Metric to monitor (for ReduceLROnPlateau)
            verbose: Print when LR changes
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose
        self.last_lr = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Step the learning rate scheduler"""
        # Get current learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']

        # Step scheduler
        if self.monitor is not None:
            # For schedulers that need a metric (e.g., ReduceLROnPlateau)
            metric_value = metrics.get(self.monitor)
            if metric_value is not None:
                self.scheduler.step(metric_value)
        else:
            # For schedulers that don't need a metric
            self.scheduler.step()

        # Check if LR changed
        new_lr = trainer.optimizer.param_groups[0]['lr']

        if self.verbose and new_lr != current_lr:
            print(f"Epoch {epoch}: Learning rate changed from {current_lr:.2e} to {new_lr:.2e}")


class MetricsLogger(Callback):
    """
    Log training metrics to file.
    """

    def __init__(self, log_file: str, verbose: bool = True):
        """
        Args:
            log_file: Path to log file
            verbose: Print metrics to console
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Clear log file
        with open(self.log_file, 'w') as f:
            f.write("epoch,metric,value\n")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Log metrics"""
        # Write to file
        with open(self.log_file, 'a') as f:
            for metric_name, value in metrics.items():
                f.write(f"{epoch},{metric_name},{value}\n")

        # Print to console
        if self.verbose:
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            print(f"Epoch {epoch}: {metrics_str}")


class ProgressBar(Callback):
    """
    Simple progress bar for training.
    """

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def on_epoch_begin(self, epoch: int, trainer):
        """Update progress"""
        self.current_epoch = epoch

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Print progress"""
        progress = (epoch + 1) / self.total_epochs
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)

        train_loss = metrics.get('train_loss', 0)
        val_loss = metrics.get('val_loss', 0)

        print(f"\rEpoch [{epoch+1}/{self.total_epochs}] {bar} "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f}", end='')

        if epoch + 1 == self.total_epochs:
            print()  # New line at end


class CallbackList:
    """
    Container for managing multiple callbacks.
    """

    def __init__(self, callbacks: list):
        self.callbacks = callbacks or []

    def on_train_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer):
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, epoch: int, trainer):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, trainer)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, trainer)

    def on_batch_begin(self, batch_idx: int, trainer):
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, trainer)

    def on_batch_end(self, batch_idx: int, loss: float, trainer):
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, trainer)
