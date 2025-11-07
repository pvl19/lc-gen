"""Loss functions for training autoencoders"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Mean Squared Error loss.

    Args:
        predictions: Predicted values
        targets: Target values
        reduction: 'mean', 'sum', or 'none'

    Returns:
        MSE loss
    """
    return F.mse_loss(predictions, targets, reduction=reduction)


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    MSE loss calculated only on masked positions.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Boolean tensor, True for positions to include in loss
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Masked MSE loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)

    # Extract only masked positions
    masked_predictions = predictions[mask]
    masked_targets = targets[mask]

    return F.mse_loss(masked_predictions, masked_targets, reduction=reduction)


def mae_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Mean Absolute Error loss.

    Args:
        predictions: Predicted values
        targets: Target values
        reduction: 'mean', 'sum', or 'none'

    Returns:
        MAE loss
    """
    return F.l1_loss(predictions, targets, reduction=reduction)


def masked_mae_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    MAE loss calculated only on masked positions.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Boolean tensor, True for positions to include in loss
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Masked MAE loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)

    # Extract only masked positions
    masked_predictions = predictions[mask]
    masked_targets = targets[mask]

    return F.l1_loss(masked_predictions, masked_targets, reduction=reduction)


def chi_squared_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainty: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Chi-squared loss weighted by uncertainties.

    Useful for light curves where uncertainty estimates are available.

    Args:
        predictions: Predicted flux values
        targets: Target flux values
        uncertainty: Flux uncertainties (standard deviations)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Chi-squared loss
    """
    # Avoid division by zero
    uncertainty = torch.clamp(uncertainty, min=1e-8)

    # Chi-squared: (pred - target)^2 / sigma^2
    chi2 = ((predictions - targets) ** 2) / (uncertainty ** 2)

    if reduction == 'mean':
        return chi2.mean()
    elif reduction == 'sum':
        return chi2.sum()
    else:
        return chi2


def masked_chi_squared_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainty: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Chi-squared loss calculated only on masked positions.

    Args:
        predictions: Predicted flux values
        targets: Target flux values
        uncertainty: Flux uncertainties
        mask: Boolean tensor, True for positions to include in loss
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Masked chi-squared loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)

    # Flatten tensors for easier indexing
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)
    uncertainty_flat = uncertainty.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Extract only masked positions
    masked_pred = pred_flat[mask_flat]
    masked_target = target_flat[mask_flat]
    masked_uncertainty = uncertainty_flat[mask_flat]

    return chi_squared_loss(masked_pred, masked_target, masked_uncertainty, reduction=reduction)


def combined_chi_squared_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainty: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.8,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Combined chi-squared loss on both masked and unmasked regions.

    Useful for training on both reconstruction and infilling tasks.

    Args:
        predictions: Predicted flux values
        targets: Target flux values
        uncertainty: Flux uncertainties
        mask: Boolean tensor, True for masked positions
        alpha: Weight for masked loss (1-alpha for unmasked loss)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Combined chi-squared loss
    """
    # Flatten tensors
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)
    uncertainty_flat = uncertainty.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Masked loss
    if mask_flat.sum() > 0:
        masked_pred = pred_flat[mask_flat]
        masked_target = target_flat[mask_flat]
        masked_uncertainty = uncertainty_flat[mask_flat]
        masked_loss = chi_squared_loss(masked_pred, masked_target, masked_uncertainty, reduction=reduction)
    else:
        masked_loss = torch.tensor(0.0, device=predictions.device)

    # Unmasked loss
    if (~mask_flat).sum() > 0:
        unmasked_pred = pred_flat[~mask_flat]
        unmasked_target = target_flat[~mask_flat]
        unmasked_uncertainty = uncertainty_flat[~mask_flat]
        unmasked_loss = chi_squared_loss(unmasked_pred, unmasked_target, unmasked_uncertainty, reduction=reduction)
    else:
        unmasked_loss = torch.tensor(0.0, device=predictions.device)

    # Weighted combination
    return alpha * masked_loss + (1 - alpha) * unmasked_loss


class MaskedReconstructionLoss(nn.Module):
    """
    Loss module for masked reconstruction.

    Can use different loss types (MSE, MAE, chi-squared) and computes
    loss only on masked positions.
    """

    def __init__(self, loss_type: str = 'mse'):
        """
        Args:
            loss_type: 'mse', 'mae', or 'chi_squared'
        """
        super().__init__()
        self.loss_type = loss_type.lower()

        if self.loss_type not in ['mse', 'mae', 'chi_squared']:
            raise ValueError(f"Unknown loss type: {loss_type}. Choose from ['mse', 'mae', 'chi_squared']")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            predictions: Predicted values
            targets: Target values
            mask: Optional boolean mask (True for masked positions)
            uncertainty: Optional uncertainties (for chi-squared loss)

        Returns:
            Loss value
        """
        if mask is None:
            # No masking, compute loss on everything
            if self.loss_type == 'mse':
                return mse_loss(predictions, targets)
            elif self.loss_type == 'mae':
                return mae_loss(predictions, targets)
            elif self.loss_type == 'chi_squared':
                if uncertainty is None:
                    raise ValueError("Chi-squared loss requires uncertainty estimates")
                return chi_squared_loss(predictions, targets, uncertainty)
        else:
            # Masked loss
            if self.loss_type == 'mse':
                return masked_mse_loss(predictions, targets, mask)
            elif self.loss_type == 'mae':
                return masked_mae_loss(predictions, targets, mask)
            elif self.loss_type == 'chi_squared':
                if uncertainty is None:
                    raise ValueError("Chi-squared loss requires uncertainty estimates")
                return masked_chi_squared_loss(predictions, targets, uncertainty, mask)


# Loss function registry for easy configuration
LOSS_REGISTRY = {
    'mse': mse_loss,
    'masked_mse': masked_mse_loss,
    'mae': mae_loss,
    'masked_mae': masked_mae_loss,
    'chi_squared': chi_squared_loss,
    'masked_chi_squared': masked_chi_squared_loss,
    'combined_chi_squared': combined_chi_squared_loss,
}


def get_loss_fn(loss_name: str):
    """
    Get loss function by name.

    Args:
        loss_name: Name of loss function

    Returns:
        Loss function

    Raises:
        ValueError: If loss_name not found in registry
    """
    if loss_name.lower() not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Choose from {list(LOSS_REGISTRY.keys())}"
        )

    return LOSS_REGISTRY[loss_name.lower()]
