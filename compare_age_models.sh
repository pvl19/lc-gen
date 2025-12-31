#!/bin/bash
# Compare age predictions: Latent Space Model vs Gyro Baseline
#
# Usage: ./compare_age_models.sh [version]
# Example: ./compare_age_models.sh v1

VERSION=${1:-"default"}
OUTPUT_DIR="output/age_predictor/comparison_${VERSION}"

echo "Running age model comparison with version: ${VERSION}"
echo "Output directory: ${OUTPUT_DIR}"

python scripts/compare_age_models.py \
  --model_path output/simple_rnn/models/baseline_v15_masking_real_60e.pt \
  --h5_path data/timeseries.h5 \
  --pickle_path data/star_sector_lc_formatted.pickle \
  --age_csv data/TIC_cf_data.csv \
  --output_dir "${OUTPUT_DIR}" \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --max_length 2048 \
  --pooling_mode mean \
  --hidden_dims 128 64 32 \
  --dropout 0.2 \
  --lr 1e-3 \
  --n_epochs 100 \
  --batch_size 64 \
  --n_folds 10 \
  --seed 42
