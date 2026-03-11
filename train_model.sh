#!/usr/bin/env bash
# Local training script with checkpoint resume support
#
# To resume from a previous checkpoint:
#   1. Set RESUME_FROM variable below to point to your checkpoint
#   2. Update --epochs to your new target (e.g., 50 or 100)
#   3. Training will continue from where it left off
#
# Example: RESUME_FROM="output/my_run/model.pt"
# Leave empty ("") for fresh training
RESUME_FROM=""

#### Build resume flag if checkpoint specified
RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_FLAG="--resume_from $RESUME_FROM"
    echo "Resuming from checkpoint: $RESUME_FROM"
else
    echo "Starting fresh training (no checkpoint specified)"
fi

python3 src/lcgen/train_simple_rnn.py \
    --input data/timeseries_x.h5 \
    --output_dir output \
    --random_seed 19 \
    --direction bi \
    --max_length 2048 \
    --num_samples 1024 \
    --batch_size 256 \
    --hidden_size 64 \
    --output_name model.pt \
    --epochs 3 \
    --lr 1e-3 \
    --min_size 5 \
    --max_size 720 \
    --mask_portion 0.4 \
    --use_flow \
    --mode parallel \
    --val_split 0.1 \
    --val_k_values 1,2,4,8 \
    --K 128 \
    --k_spacing log \
    --patience 10 \
    --min_delta 0.0 \
    --save_every 1 \
    $RESUME_FLAG
