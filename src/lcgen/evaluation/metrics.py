"""Evaluation metrics for model performance"""

import torch
import numpy as np
from typing import Union, Optional


def compute_mse(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Compute Mean Squared Error.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional boolean mask (True for positions to include)

    Returns:
        MSE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    return np.mean((predictions - targets) ** 2)


def compute_mae(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional boolean mask (True for positions to include)

    Returns:
        MAE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    return np.mean(np.abs(predictions - targets))


def compute_rmse(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional boolean mask (True for positions to include)

    Returns:
        RMSE value
    """
    mse = compute_mse(predictions, targets, mask)
    return np.sqrt(mse)


def compute_chi_squared(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    uncertainty: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Compute chi-squared statistic.

    Args:
        predictions: Predicted values
        targets: Target values
        uncertainty: Uncertainty estimates (standard deviations)
        mask: Optional boolean mask (True for positions to include)

    Returns:
        Chi-squared value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
        uncertainty = uncertainty[mask]

    # Avoid division by zero
    uncertainty = np.maximum(uncertainty, 1e-8)

    chi2 = np.sum(((predictions - targets) ** 2) / (uncertainty ** 2))
    return chi2


def compute_reduced_chi_squared(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    uncertainty: Union[torch.Tensor, np.ndarray],
    n_params: int = 0,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Compute reduced chi-squared statistic.

    Args:
        predictions: Predicted values
        targets: Target values
        uncertainty: Uncertainty estimates
        n_params: Number of model parameters (degrees of freedom)
        mask: Optional boolean mask (True for positions to include)

    Returns:
        Reduced chi-squared value
    """
    chi2 = compute_chi_squared(predictions, targets, uncertainty, mask)

    # Convert to numpy for shape access
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        n_points = np.sum(mask)
    else:
        n_points = predictions.size

    dof = n_points - n_params
    if dof <= 0:
        return float('inf')

    return chi2 / dof


def compute_r2_score(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Compute R^2 (coefficient of determination) score.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional boolean mask (True for positions to include)

    Returns:
        R^2 score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def compute_correlation(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional boolean mask (True for positions to include)

    Returns:
        Correlation coefficient
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    return np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]


def evaluate_reconstruction(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    uncertainty: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> dict:
    """
    Compute multiple evaluation metrics for reconstruction quality.

    Args:
        predictions: Predicted values
        targets: Target values
        uncertainty: Optional uncertainty estimates
        mask: Optional boolean mask

    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'mse': compute_mse(predictions, targets, mask),
        'mae': compute_mae(predictions, targets, mask),
        'rmse': compute_rmse(predictions, targets, mask),
        'r2': compute_r2_score(predictions, targets, mask),
        'correlation': compute_correlation(predictions, targets, mask),
    }

    if uncertainty is not None:
        metrics['chi_squared'] = compute_chi_squared(predictions, targets, uncertainty, mask)
        metrics['reduced_chi_squared'] = compute_reduced_chi_squared(
            predictions, targets, uncertainty, mask=mask
        )

    return metrics
