#!/usr/bin/env python
"""
K-fold cross-validation for age prediction from metadata.

Each fold trains a fresh model and evaluates on held-out data.
This ensures every sample gets an unbiased prediction from a model that never saw it.

Supports two encoder architectures:
  - mlp: Simple MLP encoder (default)
  - transformer: Transformer encoder with self-attention

Usage:
    # MLP encoder (default)
    python scripts/kfold_metadata_age.py --h5_path data/timeseries_x.h5 --output_dir output/kfold_metadata_age

    # Transformer encoder
    python scripts/kfold_metadata_age.py --model_type transformer --output_dir output/kfold_transformer_age

"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lcgen.models.MetadataAgePredictor import (
    MetadataAgePredictor,
    MetadataStandardizer,
    DEFAULT_METADATA_FIELDS,
)
from lcgen.models.TransformerAgePredictor import TransformerAgePredictor


def load_data(h5_path: str) -> dict:
    """Load metadata from h5 file."""
    data = {}
    with h5py.File(h5_path, "r") as f:
        metadata_group = f["metadata"]
        for key in metadata_group.keys():
            arr = metadata_group[key][:]
            if arr.dtype.kind == "S":
                arr = np.array([x.decode("utf-8") if isinstance(x, bytes) else x for x in arr])
            data[key] = arr
    return data


def get_valid_indices(data: dict) -> np.ndarray:
    """Get indices of samples with valid Prot and age."""
    prot = np.array(data["Prot"], dtype=np.float32)
    age = np.array(data["age_Myr"], dtype=np.float32)
    valid_mask = ~np.isnan(prot) & ~np.isnan(age) & (prot > 0) & (age > 0)
    return np.where(valid_mask)[0]


def prepare_fold_data(
    data: dict,
    indices: np.ndarray,
    standardizer: MetadataStandardizer,
    device: torch.device,
) -> tuple:
    """Prepare tensors for a fold.

    Args:
        data: full dataset dict
        indices: indices to use for this split
        standardizer: MetadataStandardizer instance (uses fixed transforms, no fitting)
        device: torch device

    Returns:
        metadata_tensor, mask_tensor, prot_tensor, age_tensor
    """
    # Subset the data
    subset = {}
    for field in DEFAULT_METADATA_FIELDS + ["Prot", "age_Myr"]:
        subset[field] = np.array(data[field])[indices]

    # Transform using fixed rules and get validity mask
    metadata_features, mask = standardizer.transform(subset, return_mask=True)

    # Get Prot and age (already filtered to valid)
    prot = np.array(subset["Prot"], dtype=np.float32)
    age = np.array(subset["age_Myr"], dtype=np.float32)

    # Log10 transform
    log_prot = np.log10(prot)
    log_age = np.log10(age)

    # Convert to tensors
    metadata_tensor = torch.tensor(metadata_features, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
    prot_tensor = torch.tensor(log_prot, dtype=torch.float32, device=device)
    age_tensor = torch.tensor(log_age, dtype=torch.float32, device=device)

    return metadata_tensor, mask_tensor, prot_tensor, age_tensor


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for metadata, mask, prot, age in dataloader:
        optimizer.zero_grad()
        output = model(metadata, prot, age, mask=mask)
        loss = output["nll"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate_epoch(model, dataloader, device):
    """Compute validation NLL."""
    model.eval()
    total_nll = 0.0
    n_batches = 0

    with torch.no_grad():
        for metadata, mask, prot, age in dataloader:
            output = model(metadata, prot, age, mask=mask)
            total_nll += output["nll"].item()
            n_batches += 1

    return total_nll / n_batches


def predict_fold(model, dataloader, device, n_samples=100):
    """Get predictions for a fold."""
    model.eval()
    all_pred_mean = []
    all_pred_std = []
    all_true = []

    with torch.no_grad():
        for metadata, mask, prot, age in dataloader:
            pred = model.predict(metadata, prot, n_samples=n_samples, mask=mask)
            all_pred_mean.append(pred["log_age_mean"].cpu().numpy())
            all_pred_std.append(pred["log_age_std"].cpu().numpy())
            all_true.append(age.cpu().numpy())

    return (
        np.concatenate(all_pred_mean),
        np.concatenate(all_pred_std),
        np.concatenate(all_true),
    )


def create_model(args, device):
    """Create model based on model_type argument."""
    if args.model_type == "mlp":
        model = MetadataAgePredictor(
            n_metadata_features=len(DEFAULT_METADATA_FIELDS),
            latent_dim=args.latent_dim,
            encoder_hidden_dims=args.encoder_hidden,
            encoder_dropout=args.encoder_dropout,
            flow_transforms=args.flow_transforms,
            flow_hidden_features=args.flow_hidden,
            use_mask=args.use_mask,
            use_random_masking=args.use_random_masking,
            mask_poisson_lambda=args.mask_poisson_lambda,
        )
    elif args.model_type == "transformer":
        model = TransformerAgePredictor(
            n_metadata_features=len(DEFAULT_METADATA_FIELDS),
            latent_dim=args.latent_dim,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            encoder_dropout=args.encoder_dropout,
            flow_transforms=args.flow_transforms,
            flow_hidden_features=args.flow_hidden,
            use_random_masking=args.use_random_masking,
            mask_poisson_lambda=args.mask_poisson_lambda,
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    return model.to(device)


def train_fold(
    train_loader,
    val_loader,
    args,
    device,
    fold_idx,
):
    """Train model for one fold."""
    model = create_model(args, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_nll = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        train_nll = train_epoch(model, train_loader, optimizer, device)
        val_nll = validate_epoch(model, val_loader, device)
        scheduler.step(val_nll)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Fold {fold_idx+1} Epoch {epoch+1}: train={train_nll:.4f}, val={val_nll:.4f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, best_val_nll


def compute_metrics(pred_log_age, true_log_age, pred_std=None):
    """Compute evaluation metrics."""
    # Log-space metrics
    errors_log = pred_log_age - true_log_age
    mse_log = np.mean(errors_log ** 2)
    mae_log = np.mean(np.abs(errors_log))
    rmse_log = np.sqrt(mse_log)

    # Linear-space metrics (convert from log10)
    pred_age = 10 ** pred_log_age
    true_age = 10 ** true_log_age
    errors_linear = pred_age - true_age

    mae_myr = np.mean(np.abs(errors_linear))
    mape = np.mean(np.abs(errors_linear) / true_age) * 100
    median_ape = np.median(np.abs(errors_linear) / true_age) * 100

    # Correlation
    correlation = np.corrcoef(pred_log_age, true_log_age)[0, 1]

    metrics = {
        "mse_log": float(mse_log),
        "mae_log": float(mae_log),
        "rmse_log": float(rmse_log),
        "mae_myr": float(mae_myr),
        "mape": float(mape),
        "median_ape": float(median_ape),
        "correlation": float(correlation),
    }

    # Calibration metrics if uncertainty available
    if pred_std is not None:
        z_scores = np.abs(errors_log) / pred_std
        within_1sigma = np.mean(z_scores < 1.0) * 100
        within_2sigma = np.mean(z_scores < 2.0) * 100
        metrics["within_1sigma"] = float(within_1sigma)
        metrics["within_2sigma"] = float(within_2sigma)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="K-fold CV for metadata age prediction")
    parser.add_argument("--h5_path", type=str, default="data/timeseries_x.h5")
    parser.add_argument("--output_dir", type=str, default="output/kfold_metadata_age")
    parser.add_argument("--n_folds", type=int, default=10)

    # Model type selection
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "transformer"],
                        help="Encoder architecture: 'mlp' or 'transformer'")

    # Common encoder arguments
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)

    # MLP-specific arguments
    parser.add_argument("--encoder_hidden", type=int, nargs="+", default=[64, 64],
                        help="MLP hidden layer sizes (only used with --model_type mlp)")

    # Transformer-specific arguments
    parser.add_argument("--d_model", type=int, default=64,
                        help="Transformer embedding dimension (only used with --model_type transformer)")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads (only used with --model_type transformer)")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of transformer layers (only used with --model_type transformer)")
    parser.add_argument("--d_ff", type=int, default=128,
                        help="Feed-forward hidden dimension (only used with --model_type transformer)")

    # Flow arguments
    parser.add_argument("--flow_transforms", type=int, default=4)
    parser.add_argument("--flow_hidden", type=int, nargs="+", default=[64, 64])

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_mc_samples", type=int, default=100, help="Monte Carlo samples for prediction")

    # Masking arguments
    parser.add_argument("--use_mask", action="store_true",
                        help="Use mask input for MLP encoder (concatenates mask to input)")
    parser.add_argument("--use_random_masking", action="store_true",
                        help="Randomly mask features during training for robustness")
    parser.add_argument("--mask_poisson_lambda", type=float, default=2.0,
                        help="Mean of Poisson distribution for number of features to randomly mask")
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.h5_path}...")
    data = load_data(args.h5_path)
    print(f"Loaded {len(data['tic'])} records")

    # Get valid indices
    valid_indices = get_valid_indices(data)
    n_valid = len(valid_indices)
    print(f"Valid samples with Prot and age: {n_valid}")

    # Store all predictions
    all_predictions = {
        "indices": valid_indices.tolist(),
        "tic": [int(data["tic"][i]) for i in valid_indices],
        "sector": [int(data["sector"][i]) for i in valid_indices],
        "true_log_age": [],
        "pred_log_age": [],
        "pred_std": [],
        "fold": [],
    }

    # K-fold cross-validation
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []

    # Print model info
    temp_model = create_model(args, device)
    n_params = sum(p.numel() for p in temp_model.parameters())
    print(f"\nModel type: {args.model_type}")
    print(f"Model parameters: {n_params:,}")
    if args.model_type == "transformer":
        print(f"  d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}, d_ff={args.d_ff}")
    else:
        print(f"  encoder_hidden={args.encoder_hidden}")
        if args.use_mask:
            print(f"  use_mask=True (input dimension doubled)")
    if args.use_random_masking:
        print(f"  random_masking: Poisson(λ={args.mask_poisson_lambda})")
    del temp_model

    print(f"\nStarting {args.n_folds}-fold cross-validation...")

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(valid_indices)):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{args.n_folds}")
        print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

        # Get actual data indices
        train_data_idx = valid_indices[train_idx]
        test_data_idx = valid_indices[test_idx]

        # Split train into train/val for early stopping
        n_train = len(train_data_idx)
        n_val = int(n_train * 0.1)
        perm = np.random.permutation(n_train)
        val_subset_idx = train_data_idx[perm[:n_val]]
        train_subset_idx = train_data_idx[perm[n_val:]]

        # Create standardizer (fit on train only)
        standardizer = MetadataStandardizer(fields=DEFAULT_METADATA_FIELDS)

        # Prepare data (returns metadata, mask, prot, age tensors)
        train_meta, train_mask, train_prot, train_age = prepare_fold_data(
            data, train_subset_idx, standardizer, device
        )
        val_meta, val_mask, val_prot, val_age = prepare_fold_data(
            data, val_subset_idx, standardizer, device
        )
        test_meta, test_mask, test_prot, test_age = prepare_fold_data(
            data, test_data_idx, standardizer, device
        )

        # Create dataloaders (include mask for masking support)
        train_loader = DataLoader(
            TensorDataset(train_meta, train_mask, train_prot, train_age),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_meta, val_mask, val_prot, val_age),
            batch_size=args.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(test_meta, test_mask, test_prot, test_age),
            batch_size=args.batch_size, shuffle=False
        )

        # Train model
        model, best_val_nll = train_fold(
            train_loader, val_loader, args, device, fold_idx
        )

        # Get test predictions
        pred_mean, pred_std, true_log_age = predict_fold(
            model, test_loader, device, n_samples=args.n_mc_samples
        )

        # Store predictions
        all_predictions["true_log_age"].extend(true_log_age.tolist())
        all_predictions["pred_log_age"].extend(pred_mean.tolist())
        all_predictions["pred_std"].extend(pred_std.tolist())
        all_predictions["fold"].extend([fold_idx] * len(test_idx))

        # Compute fold metrics
        metrics = compute_metrics(pred_mean, true_log_age, pred_std)
        metrics["fold"] = fold_idx
        metrics["val_nll"] = best_val_nll
        fold_metrics.append(metrics)

        print(f"  Test metrics:")
        print(f"    MAE (log): {metrics['mae_log']:.4f}")
        print(f"    MAPE: {metrics['mape']:.1f}%")
        print(f"    Correlation: {metrics['correlation']:.4f}")
        if "within_1sigma" in metrics:
            print(f"    Within 1σ: {metrics['within_1sigma']:.1f}%")

        # Save fold model
        torch.save({
            "model_state_dict": model.state_dict(),
            "standardizer_state": standardizer.state_dict(),
            "fold": fold_idx,
            "metrics": metrics,
        }, output_dir / f"fold_{fold_idx}_model.pt")

    # Aggregate predictions
    all_pred = np.array(all_predictions["pred_log_age"])
    all_true = np.array(all_predictions["true_log_age"])
    all_std = np.array(all_predictions["pred_std"])

    # Compute overall metrics
    overall_metrics = compute_metrics(all_pred, all_true, all_std)

    print(f"\n{'='*50}")
    print("Overall Results (all folds combined):")
    print(f"  MAE (log age): {overall_metrics['mae_log']:.4f}")
    print(f"  RMSE (log age): {overall_metrics['rmse_log']:.4f}")
    print(f"  MAE (Myr): {overall_metrics['mae_myr']:.1f}")
    print(f"  MAPE: {overall_metrics['mape']:.1f}%")
    print(f"  Median APE: {overall_metrics['median_ape']:.1f}%")
    print(f"  Correlation: {overall_metrics['correlation']:.4f}")
    if "within_1sigma" in overall_metrics:
        print(f"  Within 1σ: {overall_metrics['within_1sigma']:.1f}%")
        print(f"  Within 2σ: {overall_metrics['within_2sigma']:.1f}%")

    # Save predictions
    np.save(output_dir / "predictions.npy", {
        "pred_log_age": all_pred,
        "true_log_age": all_true,
        "pred_std": all_std,
        "tic": all_predictions["tic"],
        "sector": all_predictions["sector"],
        "fold": all_predictions["fold"],
    })

    # Save metrics
    results = {
        "args": vars(args),
        "fold_metrics": fold_metrics,
        "overall_metrics": overall_metrics,
        "n_samples": n_valid,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    # Generate summary plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Predicted vs True
        ax = axes[0]
        true_age_myr = 10 ** all_true
        pred_age_myr = 10 ** all_pred
        ax.scatter(true_age_myr, pred_age_myr, alpha=0.3, s=10)
        lims = [min(true_age_myr.min(), pred_age_myr.min()),
                max(true_age_myr.max(), pred_age_myr.max())]
        ax.plot(lims, lims, 'r--', label='y=x')
        ax.set_xlabel("True Age (Myr)")
        ax.set_ylabel("Predicted Age (Myr)")
        ax.set_title(f"Age Prediction (r={overall_metrics['correlation']:.3f})")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()

        # Residuals
        ax = axes[1]
        residuals = all_pred - all_true
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel("Residual (log age)")
        ax.set_ylabel("Count")
        ax.set_title(f"Residuals (MAE={overall_metrics['mae_log']:.3f})")

        # Calibration
        if all_std is not None:
            ax = axes[2]
            z_scores = np.abs(residuals) / all_std
            ax.hist(z_scores, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(1.0, color='red', linestyle='--', label='1σ')
            ax.axvline(2.0, color='orange', linestyle='--', label='2σ')
            ax.set_xlabel("|z-score|")
            ax.set_ylabel("Count")
            ax.set_title(f"Calibration (1σ: {overall_metrics['within_1sigma']:.0f}%)")
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "kfold_results.png", dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_dir / 'kfold_results.png'}")

    except ImportError:
        print("matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    main()
