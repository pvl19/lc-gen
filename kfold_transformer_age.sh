#!/bin/bash
# K-fold cross-validation for age prediction using Transformer encoder
#
# Usage: ./kfold_transformer_age.sh [output_dir] [h5_path]
# Example: ./kfold_transformer_age.sh output/kfold_transformer_age data/timeseries_x.h5

OUTPUT_DIR=${1:-"output/metadata_age_inference/transformer/masking/lambda_1"}
H5_PATH=${2:-"data/timeseries_x.h5"}

echo "Running k-fold age prediction with Transformer encoder:"
echo "  H5 file: ${H5_PATH}"
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

python scripts/kfold_metadata_age.py \
  --h5_path ${H5_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model_type transformer \
  --n_folds 10 \
  --latent_dim 32 \
  --d_model 64 \
  --n_heads 4 \
  --n_layers 2 \
  --d_ff 128 \
  --encoder_dropout 0.1 \
  --flow_transforms 4 \
  --flow_hidden 64 64 \
  --batch_size 64 \
  --lr 1e-3 \
  --epochs 200 \
  --patience 15 \
  --seed 42 \
  --n_mc_samples 100 \
  --use_random_masking \
  --mask_poisson_lambda 1.0
