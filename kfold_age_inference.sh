#!/bin/bash
# K-fold cross-validation for age prediction from latent space + BPRP0
# Each star gets a prediction from a model that never saw it during training
#
# Usage: ./kfold_age_inference.sh [version] [pooling_mode]
# Example: ./kfold_age_inference.sh v1 mean
# Example: ./kfold_age_inference.sh v1 multiscale
# Example: ./kfold_age_inference.sh v1 final
#
# Pooling modes:
#   mean       - Mean pooling over valid timesteps (128 dims for hidden_size=64 bi)
#   multiscale - Multi-scale features: mean, max, std, quartiles, chunks (1536 dims)
#   final      - Final hidden state only (128 dims)

VERSION=${1:-"default"}
POOLING_MODE=${2:-"mean"}
OUTPUT_DIR="output/age_predictor/kfold_${VERSION}_${POOLING_MODE}"

echo "Running k-fold age inference:"
echo "  Version: ${VERSION}"
echo "  Pooling mode: ${POOLING_MODE}"
echo "  Output directory: ${OUTPUT_DIR}"

python scripts/kfold_age_inference.py \
  --model_path output/simple_rnn/models/baseline_v17_real_final.pt \
  --h5_path data/timeseries.h5 \
  --pickle_path data/star_sector_lc_formatted.pickle \
  --age_csv data/TIC_cf_data.csv \
  --output_dir "${OUTPUT_DIR}" \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --max_length 2048 \
  --pooling_mode "${POOLING_MODE}" \
  --stratify_by_age \
  --n_age_bins 20 \
  --n_folds 10 \
  --hidden_dims 128 64 32 \
  --dropout 0.2 \
  --lr 1e-3 \
  --n_epochs 100 \
  --batch_size 64 \
  --seed 42
