#!/usr/bin/env python
"""
Train the MetadataAgePredictor model.

This script trains an MLP encoder + normalizing flow model to predict stellar age
from metadata features, conditioned on rotation period (Prot).

Usage:
    python scripts/train_metadata_age_predictor.py --h5_path data/timeseries_x.h5 --output_dir output/metadata_age

"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lcgen.models.MetadataAgePredictor import (
    MetadataAgePredictor,
    MetadataStandardizer,
    DEFAULT_METADATA_FIELDS,
)


def load_data(h5_path: str) -> dict:
    """Load metadata from h5 file."""
    data = {}
    with h5py.File(h5_path, "r") as f:
        metadata_group = f["metadata"]
        for key in metadata_group.keys():
            arr = metadata_group[key][:]
            # Decode bytes if needed
            if arr.dtype.kind == "S":
                arr = np.array([x.decode("utf-8") if isinstance(x, bytes) else x for x in arr])
            data[key] = arr
    return data


def prepare_dataset(
    data: dict,
    standardizer: MetadataStandardizer,
    device: torch.device,
) -> tuple:
    """Prepare tensors for training.

    Returns:
        metadata_tensor, mask_tensor, prot_tensor, age_tensor, valid_indices
    """
    n_samples = len(data["tic"])

    # Get normalized metadata features and validity mask (fixed transformations)
    metadata_features, feature_mask = standardizer.transform(data, return_mask=True)

    # Get Prot and age
    prot = np.array(data["Prot"], dtype=np.float32)
    age = np.array(data["age_Myr"], dtype=np.float32)

    # Create valid mask: need valid Prot and age
    valid_mask = ~np.isnan(prot) & ~np.isnan(age) & (prot > 0) & (age > 0)
    print(f"Valid samples with Prot and age: {valid_mask.sum()}/{n_samples}")

    # Filter to valid samples
    metadata_features = metadata_features[valid_mask]
    feature_mask = feature_mask[valid_mask]
    prot = prot[valid_mask]
    age = age[valid_mask]

    # Log transform Prot and age
    log_prot = np.log10(prot)
    log_age = np.log10(age)

    # Convert to tensors
    metadata_tensor = torch.tensor(metadata_features, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(feature_mask, dtype=torch.float32, device=device)
    prot_tensor = torch.tensor(log_prot, dtype=torch.float32, device=device)
    age_tensor = torch.tensor(log_age, dtype=torch.float32, device=device)

    return metadata_tensor, mask_tensor, prot_tensor, age_tensor, valid_mask


def train_epoch(
    model: MetadataAgePredictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
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


def validate(
    model: MetadataAgePredictor,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()
    total_nll = 0.0
    all_pred_log_age = []
    all_true_log_age = []
    n_batches = 0

    with torch.no_grad():
        for metadata, mask, prot, age in dataloader:
            output = model(metadata, prot, age, mask=mask)
            total_nll += output["nll"].item()

            # Get predictions
            pred = model.predict(metadata, prot, n_samples=100, mask=mask)
            all_pred_log_age.append(pred["log_age_mean"].cpu())
            all_true_log_age.append(age.cpu())
            n_batches += 1

    pred_log_age = torch.cat(all_pred_log_age)
    true_log_age = torch.cat(all_true_log_age)

    # Compute metrics
    mse = ((pred_log_age - true_log_age) ** 2).mean().item()
    mae = (pred_log_age - true_log_age).abs().mean().item()

    # Convert to linear age for interpretability
    pred_age = 10 ** pred_log_age
    true_age = 10 ** true_log_age
    mae_myr = (pred_age - true_age).abs().mean().item()
    mape = ((pred_age - true_age).abs() / true_age).mean().item() * 100

    return {
        "nll": total_nll / n_batches,
        "mse_log": mse,
        "mae_log": mae,
        "mae_myr": mae_myr,
        "mape": mape,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MetadataAgePredictor")
    parser.add_argument("--h5_path", type=str, default="data/timeseries_x.h5")
    parser.add_argument("--output_dir", type=str, default="output/metadata_age_predictor")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--encoder_hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--flow_transforms", type=int, default=4)
    parser.add_argument("--flow_hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Masking arguments
    parser.add_argument("--use_mask", action="store_true",
                        help="Use mask input for encoder (concatenates mask to input)")
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

    # Prepare standardizer
    standardizer = MetadataStandardizer(fields=DEFAULT_METADATA_FIELDS)

    # Prepare dataset
    metadata_tensor, mask_tensor, prot_tensor, age_tensor, valid_mask = prepare_dataset(
        data, standardizer, device
    )
    n_valid = len(age_tensor)
    print(f"Prepared {n_valid} valid samples")

    # Train/val split
    n_val = int(n_valid * args.val_split)
    n_train = n_valid - n_val

    indices = torch.randperm(n_valid)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = TensorDataset(
        metadata_tensor[train_idx],
        mask_tensor[train_idx],
        prot_tensor[train_idx],
        age_tensor[train_idx],
    )
    val_dataset = TensorDataset(
        metadata_tensor[val_idx],
        mask_tensor[val_idx],
        prot_tensor[val_idx],
        age_tensor[val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {n_train}, Val: {n_val}")

    # Create model
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
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.use_mask:
        print(f"  use_mask=True (input dimension doubled)")
    if args.use_random_masking:
        print(f"  random_masking: Poisson(λ={args.mask_poisson_lambda})")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_nll = float("inf")
    patience_counter = 0
    history = {"train_nll": [], "val_nll": [], "val_mae_log": [], "val_mape": []}

    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_nll = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)

        scheduler.step(val_metrics["nll"])

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_metrics["nll"])
        history["val_mae_log"].append(val_metrics["mae_log"])
        history["val_mape"].append(val_metrics["mape"])

        print(
            f"Epoch {epoch+1:3d} | "
            f"Train NLL: {train_nll:.4f} | "
            f"Val NLL: {val_metrics['nll']:.4f} | "
            f"Val MAE(log): {val_metrics['mae_log']:.4f} | "
            f"Val MAPE: {val_metrics['mape']:.1f}%"
        )

        # Early stopping
        if val_metrics["nll"] < best_val_nll:
            best_val_nll = val_metrics["nll"]
            patience_counter = 0

            # Save best model
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "standardizer_state": standardizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "args": vars(args),
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save final model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "standardizer_state": standardizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
    }
    torch.save(checkpoint, output_dir / "final_model.pt")

    print(f"\nTraining complete. Best val NLL: {best_val_nll:.4f}")
    print(f"Models saved to {output_dir}")


if __name__ == "__main__":
    main()
