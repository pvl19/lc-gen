"""Visualization utilities for model outputs"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Union, Tuple
from pathlib import Path


def plot_reconstruction(
    original: Union[torch.Tensor, np.ndarray],
    reconstructed: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Reconstruction",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Plot original vs reconstructed data with optional masking visualization.

    Args:
        original: Original data
        reconstructed: Reconstructed data
        mask: Optional boolean mask (True for masked regions)
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if needed
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot reconstruction
    x_axis = np.arange(len(original))
    axes[0].plot(x_axis, original, label='Original', alpha=0.7, linewidth=2)
    axes[0].plot(x_axis, reconstructed, label='Reconstructed', alpha=0.7, linewidth=2)

    # Highlight masked regions if provided
    if mask is not None:
        axes[0].fill_between(
            x_axis,
            original.min(),
            original.max(),
            where=mask,
            color="lightgrey",
            alpha=0.5,
            label='Masked Region'
        )

    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'{title} - Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot residuals
    residual = original - reconstructed
    axes[1].plot(x_axis, residual, alpha=0.7, linewidth=2, color='C2')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Residual')
    axes[1].set_title(f'{title} - Residuals')
    axes[1].grid(True, alpha=0.3)

    # Add MSE to residual plot
    mse = np.mean((original - reconstructed) ** 2)
    axes[1].text(
        0.02, 0.98,
        f'MSE: {mse:.6f}',
        transform=axes[1].transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    learning_rates: Optional[list] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot training and validation losses over epochs.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        learning_rates: Optional list of learning rates
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display the plot
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    n_plots = 2 if learning_rates is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label='Train Loss', color='black', linestyle='dashed', linewidth=2)
    axes[0].plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)

    # Annotate best validation loss
    best_val_loss = min(val_losses)
    best_epoch = val_losses.index(best_val_loss) + 1
    axes[0].axhline(y=best_val_loss, color='red', linestyle=':', alpha=0.5)
    axes[0].text(
        len(epochs) / 2,
        (max(val_losses) + best_val_loss) / 2,
        f'Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch})',
        fontsize=10,
        ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot learning rate if provided
    if learning_rates is not None:
        axes[1].plot(epochs, learning_rates, color='blue', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title(f'{title} - Learning Rate')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_latent_space(
    latent_vectors: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    method: str = 'pca',
    title: str = "Latent Space Visualization",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Visualize latent space using dimensionality reduction.

    Args:
        latent_vectors: Latent representations (N, latent_dim)
        labels: Optional labels for coloring points
        method: 'pca' or 'tsne'
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display the plot
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.detach().cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Apply dimensionality reduction
    if method.lower() == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(latent_vectors)
        method_name = "PCA"
    elif method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(latent_vectors)
        method_name = "t-SNE"
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'pca' or 'tsne'")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6)

    ax.set_xlabel(f'{method_name} Component 1')
    ax.set_ylabel(f'{method_name} Component 2')
    ax.set_title(f'{title} ({method_name})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_light_curve(
    flux: Union[torch.Tensor, np.ndarray],
    time: Optional[Union[torch.Tensor, np.ndarray]] = None,
    flux_err: Optional[Union[torch.Tensor, np.ndarray]] = None,
    flux_pred: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Light Curve",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot light curve with optional predictions, uncertainties, and masking.

    Args:
        flux: Flux values
        time: Optional time values (x-axis)
        flux_err: Optional flux uncertainties
        flux_pred: Optional predicted flux values
        mask: Optional boolean mask (True for masked regions)
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display the plot
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy
    if isinstance(flux, torch.Tensor):
        flux = flux.detach().cpu().numpy()
    if time is not None and isinstance(time, torch.Tensor):
        time = time.detach().cpu().numpy()
    if flux_err is not None and isinstance(flux_err, torch.Tensor):
        flux_err = flux_err.detach().cpu().numpy()
    if flux_pred is not None and isinstance(flux_pred, torch.Tensor):
        flux_pred = flux_pred.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Default time axis
    if time is None:
        time = np.arange(len(flux))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot true flux with error bars
    if flux_err is not None:
        ax.errorbar(
            time, flux, yerr=flux_err,
            label="True Flux",
            linewidth=1,
            elinewidth=0.5,
            alpha=0.7
        )
    else:
        ax.plot(time, flux, label="True Flux", linewidth=1, alpha=0.7)

    # Plot predicted flux
    if flux_pred is not None:
        ax.plot(time, flux_pred, label="Predicted Flux", linewidth=1, alpha=0.7)

    # Highlight masked regions
    if mask is not None:
        ax.fill_between(
            time,
            flux.min(),
            flux.max(),
            where=mask,
            color="lightgrey",
            alpha=0.5,
            label='Masked Region'
        )

    ax.set_xlabel('Time')
    ax.set_ylabel('Flux')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig
