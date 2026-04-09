#!/usr/bin/env bash
# Training script for PSC Bridges-2 interactive sessions
#
# Usage:
#   1. Get an interactive GPU session:
#      interact -p GPU-shared --gres=gpu:v100-32:1 -t 02:00:00 -A phy260003p
#   2. cd $HOME/lcgen
#   3. ./train_model_bridges2.sh
#
# To resume from a previous checkpoint:
#   1. Set RESUME_FROM variable below to point to your checkpoint
#   2. Update --epochs to your new target (e.g., 50 or 100)
#   3. Training will continue from where it left off
#
# Example: RESUME_FROM="output/my_run/model.pt"
# Leave empty ("") for fresh training
RESUME_FROM=""

# Define container
CONTAINER="/ocean/containers/ngc/pytorch/pytorch_24.11-py3.sif"

echo "========================================"
echo "Bridges-2 Interactive Training"
echo "========================================"

# Install dependencies (if needed)
echo "Installing dependencies..."
singularity exec --nv --bind /ocean,$HOME \
    $CONTAINER pip install --user zuko h5py 2>/dev/null || true

# Verify setup
echo "Verifying PyTorch + CUDA..."
singularity exec --nv --bind /ocean,$HOME \
    $CONTAINER python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

#### Build resume flag if checkpoint specified
RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_FLAG="--resume_from $RESUME_FROM"
    echo "Resuming from checkpoint: $RESUME_FROM"
else
    echo "Starting fresh training (no checkpoint specified)"
fi

echo "========================================"
echo "Starting training..."
echo "========================================"

# Run training with singularity container
singularity exec --nv --bind /ocean,$HOME \
    $CONTAINER python -u src/lcgen/train_simple_rnn.py \
    --input data/timeseries_x.h5 \
    --output_dir output \
    --random_seed 19 \
    --direction bi \
    --max_length 8192 \
    --num_samples 8192 \
    --batch_size 128 \
    --hidden_size 64 \
    --output_name model.pt \
    --epochs 3 \
    --lr 1e-3 \
    --min_size 5 \
    --max_size 256 \
    --mask_portion 0.4 \
    --use_flow \
    --mode parallel \
    --use_metadata \
    --conv_encoder_type unet \
    --use_conv_channels \
    --val_split 0.1 \
    --val_k_values 1,2,4,8 \
    --K 128 \
    --k_spacing log \
    --patience 10 \
    --min_delta 0.0 \
    $RESUME_FLAG

echo "========================================"
echo "Training complete!"
echo "========================================"
