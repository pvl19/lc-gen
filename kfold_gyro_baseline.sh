#!/bin/bash
# K-fold gyrochronology baseline: predict age from BPRP0 + Prot only
#
# Usage: ./kfold_gyro_baseline.sh [version]
# Example: ./kfold_gyro_baseline.sh v1

VERSION=${1:-"default"}
OUTPUT_DIR="output/age_predictor/gyro_baseline_init_slurm_comp"

echo "Running gyro baseline k-fold with version: ${VERSION}"
echo "Output directory: ${OUTPUT_DIR}"

python scripts/kfold_gyro_baseline.py \
  --age_csv data/TIC_cf_data.csv \
  --output_dir "${OUTPUT_DIR}" \
  --hidden_dims 128 64 32 \
  --dropout 0.2 \
  --lr 1e-3 \
  --n_epochs 100 \
  --batch_size 64 \
  --n_folds 10 \
  --seed 42
