#!/bin/bash
# Helper script to copy a checkpoint from your laptop to SDSC Expanse
# Run this on your LOCAL machine (Mac/laptop)

# Usage:
#   ./copy_checkpoint_to_expanse.sh <local_checkpoint_dir> <username>
#
# Example:
#   ./copy_checkpoint_to_expanse.sh output/slurm/46136857-lcgen-baseline-metadata-xchannels pvanlane

set -e  # Exit on error

if [ $# -lt 2 ]; then
    echo "ERROR: Missing arguments"
    echo ""
    echo "Usage: $0 <local_checkpoint_dir> <expanse_username>"
    echo ""
    echo "Example:"
    echo "  $0 output/slurm/46136857-lcgen-baseline-metadata-xchannels pvanlane"
    echo ""
    echo "Available local checkpoints:"
    ls -1d output/slurm/*-lcgen-* 2>/dev/null | head -5 || echo "  (none found)"
    exit 1
fi

SOURCE_DIR="$1"
USERNAME="$2"
EXPANSE_HOST="login.expanse.sdsc.edu"
REMOTE_DIR="lcgen/checkpoints/resume"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Directory not found: $SOURCE_DIR"
    exit 1
fi

# Find the .pt file
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
echo "Copying checkpoint to SDSC Expanse"
echo "========================================"
echo "Source: $SOURCE_DIR"
echo "Destination: ${USERNAME}@${EXPANSE_HOST}:~/${REMOTE_DIR}"
echo ""

# Create temporary directory with standardized names
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Preparing files..."
cp "$PT_FILE" "$TEMP_DIR/model.pt"
cp "$LOSSES_FILE" "$TEMP_DIR/losses_per_epoch.npz"
echo "  ✓ Copied $(basename $PT_FILE) → model.pt"
echo "  ✓ Copied losses_per_epoch.npz"

if [ -f "$SOURCE_DIR/train_args.json" ]; then
    cp "$SOURCE_DIR/train_args.json" "$TEMP_DIR/train_args.json"
    echo "  ✓ Copied train_args.json"
fi

# Show local checkpoint info
echo ""
echo "Local checkpoint info:"
python3 << EOF
import numpy as np
losses = np.load('$LOSSES_FILE')
n_epochs = len(losses['train_loss'])
print(f"  Epochs trained: {n_epochs}")
print(f"  Last train loss: {losses['train_loss'][-1]:.6f}")
if 'val_loss' in losses:
    print(f"  Last val loss: {losses['val_loss'][-1]:.6f}")
EOF

echo ""
echo "Copying to Expanse..."
echo "(You may be prompted for your Expanse password)"
echo ""

# Create remote directory and copy files
ssh ${USERNAME}@${EXPANSE_HOST} "mkdir -p ~/${REMOTE_DIR}"
scp $TEMP_DIR/* ${USERNAME}@${EXPANSE_HOST}:~/${REMOTE_DIR}/

echo ""
echo "========================================"
echo "✓ Copy completed successfully!"
echo "========================================"
echo ""
echo "Files copied to Expanse:"
ssh ${USERNAME}@${EXPANSE_HOST} "ls -lh ~/${REMOTE_DIR}/"

echo ""
echo "========================================"
echo "Next Steps (on Expanse)"
echo "========================================"
echo "1. SSH to Expanse:"
echo "   ssh ${USERNAME}@${EXPANSE_HOST}"
echo ""
echo "2. Edit slurm.sh and set:"
echo "   RESUME_FROM=\"checkpoints/resume/model.pt\""
echo ""
echo "3. Update --epochs to your target (e.g., 100)"
echo ""
echo "4. Submit job:"
echo "   cd ~/lcgen"
echo "   sbatch slurm.sh"
echo "========================================"
