#!/bin/bash
# Compare age predictions by loading from pre-computed CSV files (no retraining)
#
# Usage: ./compare_from_csv.sh [latent_version] [gyro_version] [output_version]
# Example: ./compare_from_csv.sh v1 v1 comparison_v1

LATENT_VERSION=${1:-"default"}
GYRO_VERSION=${2:-"default"}
OUTPUT_VERSION=${3:-"from_csv"}

LATENT_CSV="output/age_predictor/kfold_${LATENT_VERSION}/kfold_predictions.csv"
GYRO_CSV="output/age_predictor/gyro_baseline_${GYRO_VERSION}/gyro_kfold_predictions.csv"
OUTPUT_DIR="output/age_predictor/comparison_${OUTPUT_VERSION}"

echo "Comparing from CSVs:"
echo "  Latent CSV: ${LATENT_CSV}"
echo "  Gyro CSV: ${GYRO_CSV}"
echo "  Output: ${OUTPUT_DIR}"

python scripts/compare_age_models.py \
  --load_from_csv \
  --latent_csv "${LATENT_CSV}" \
  --gyro_csv "${GYRO_CSV}" \
  --output_dir "${OUTPUT_DIR}"
