#!/bin/bash
# K-fold cross-validation for age prediction from metadata
#
# Usage: ./kfold_metadata_age.sh [output_dir] [h5_path]
# Example: ./kfold_metadata_age.sh output/kfold_metadata_age data/timeseries_x.h5

OUTPUT_DIR=${1:-"output/metadata_age_inference/MLP/masking/lambda_1"}
H5_PATH=${2:-"data/timeseries_x.h5"}

echo "Running k-fold metadata age prediction:"
echo "  H5 file: ${H5_PATH}"
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

python scripts/kfold_metadata_age.py \
  --h5_path ${H5_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --n_folds 10 \
  --latent_dim 32 \
  --encoder_hidden 64 64 \
  --encoder_dropout 0.1 \
  --flow_transforms 4 \
  --flow_hidden 64 64 \
  --batch_size 64 \
  --lr 1e-3 \
  --epochs 200 \
  --patience 15 \
  --seed 42 \
  --n_mc_samples 100 \
  --use_mask \
  --use_random_masking \
  --mask_poisson_lambda 1.0
