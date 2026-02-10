#!/bin/bash
# Helper script to prepare a checkpoint for resuming training
# Run this on SDSC Expanse after a training job completes

# Usage:
#   ./setup_resume_checkpoint.sh <job_output_dir> [checkpoint_name]
#
# Examples:
#   ./setup_resume_checkpoint.sh output/46136857-lcgen-baseline-metadata-xchannels
#   ./setup_resume_checkpoint.sh output/46136857-lcgen-baseline-metadata-xchannels model_epoch20

set -e  # Exit on error

if [ $# -eq 0 ]; then
    echo "ERROR: No output directory specified"
    echo ""
    echo "Usage: $0 <job_output_dir> [checkpoint_name]"
    echo ""
    echo "Examples:"
    echo "  $0 output/46136857-lcgen-baseline-metadata-xchannels"
    echo "  $0 output/46136857-lcgen-baseline-metadata-xchannels model_epoch20"
    echo ""
    echo "Available output directories:"
    ls -1d output/*-lcgen-* 2>/dev/null | head -5 || echo "  (none found)"
    exit 1
fi

SOURCE_DIR="$1"
CHECKPOINT_NAME="${2:-model}"  # Default to 'model' if not specified

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Directory not found: $SOURCE_DIR"
    exit 1
fi

# Find the .pt file in the source directory
PT_FILE=$(find "$SOURCE_DIR" -maxdepth 1 -name "*.pt" | head -1)
if [ -z "$PT_FILE" ]; then
    echo "ERROR: No .pt checkpoint file found in $SOURCE_DIR"
    exit 1
fi

# Check for losses file
LOSSES_FILE="$SOURCE_DIR/losses_per_epoch.npz"
if [ ! -f "$LOSSES_FILE" ]; then
    echo "ERROR: losses_per_epoch.npz not found in $SOURCE_DIR"
    exit 1
fi

echo "========================================"
echo "Setting up checkpoint for resume"
echo "========================================"
echo "Source: $SOURCE_DIR"
echo "Model file: $(basename $PT_FILE)"
echo ""

# Create checkpoint directories
mkdir -p checkpoints/resume
mkdir -p checkpoints/backups

# Copy to resume directory
echo "Copying files to checkpoints/resume/..."
cp "$PT_FILE" "checkpoints/resume/${CHECKPOINT_NAME}.pt"
cp "$LOSSES_FILE" "checkpoints/resume/losses_per_epoch.npz"

# Copy train_args.json if it exists
if [ -f "$SOURCE_DIR/train_args.json" ]; then
    cp "$SOURCE_DIR/train_args.json" "checkpoints/resume/train_args.json"
    echo "  ✓ Copied train_args.json"
fi

echo "  ✓ Copied ${CHECKPOINT_NAME}.pt"
echo "  ✓ Copied losses_per_epoch.npz"

# Also create a timestamped backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="checkpoints/backups/${CHECKPOINT_NAME}_${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"
cp "$PT_FILE" "$BACKUP_DIR/${CHECKPOINT_NAME}.pt"
cp "$LOSSES_FILE" "$BACKUP_DIR/losses_per_epoch.npz"
if [ -f "$SOURCE_DIR/train_args.json" ]; then
    cp "$SOURCE_DIR/train_args.json" "$BACKUP_DIR/train_args.json"
fi
echo "  ✓ Created backup at $BACKUP_DIR"

# Show checkpoint info
echo ""
echo "========================================"
echo "Checkpoint Information"
echo "========================================"
python3 << EOF
import numpy as np
import os

losses_path = 'checkpoints/resume/losses_per_epoch.npz'
if os.path.exists(losses_path):
    losses = np.load(losses_path)
    n_epochs = len(losses['train_loss'])
    print(f"Epochs trained: {n_epochs}")
    print(f"Last train loss: {losses['train_loss'][-1]:.6f}")
    if 'val_loss' in losses:
        print(f"Last val loss: {losses['val_loss'][-1]:.6f}")
        best_val_idx = losses['val_loss'].argmin()
        print(f"Best val loss: {losses['val_loss'][best_val_idx]:.6f} at epoch {best_val_idx + 1}")
else:
    print("Could not load losses file")
EOF

echo ""
echo "========================================"
echo "Next Steps"
echo "========================================"
echo "1. Edit slurm.sh and set:"
echo "   RESUME_FROM=\"checkpoints/resume/${CHECKPOINT_NAME}.pt\""
echo ""
echo "2. Update --epochs to your target (e.g., 100)"
echo ""
echo "3. Submit job:"
echo "   sbatch slurm.sh"
echo "========================================"
