#!/bin/bash
# Predict stellar ages using saved k-fold models (no retraining)
#
# Usage for latent model:
#   ./predict_ages.sh latent [model_version] [output_name]
#   Example: ./predict_ages.sh latent v1 predictions_v1
#
# Usage for gyro model:
#   ./predict_ages.sh gyro [model_version] [output_name]
#   Example: ./predict_ages.sh gyro v1 gyro_predictions_v1

MODEL_TYPE=${1:-"latent"}
MODEL_VERSION=${2:-"default"}
OUTPUT_NAME=${3:-"predictions_${MODEL_VERSION}"}

if [ "$MODEL_TYPE" == "latent" ]; then
    MODEL_FILE="output/age_predictor/kfold_${MODEL_VERSION}/kfold_models.pt"
    OUTPUT_CSV="output/age_predictor/${OUTPUT_NAME}.csv"
    
    echo "Predicting with latent model:"
    echo "  Model file: ${MODEL_FILE}"
    echo "  Output: ${OUTPUT_CSV}"
    
    python scripts/predict_ages.py \
      --model_file "${MODEL_FILE}" \
      --model_type latent \
      --rnn_model output/simple_rnn/models/baseline_v15_masking_real_60e.pt \
      --h5_path data/timeseries.h5 \
      --pickle_path data/star_sector_lc_formatted.pickle \
      --age_csv data/TIC_cf_data.csv \
      --output_csv "${OUTPUT_CSV}" \
      --hidden_size 64 \
      --direction bi \
      --mode parallel \
      --use_flow \
      --max_length 2048 \
      --pooling_mode mean \
      --ensemble mean \
      --batch_size 32

elif [ "$MODEL_TYPE" == "gyro" ]; then
    MODEL_FILE="output/age_predictor/gyro_baseline_${MODEL_VERSION}/gyro_kfold_models.pt"
    OUTPUT_CSV="output/age_predictor/${OUTPUT_NAME}.csv"
    
    echo "Predicting with gyro model:"
    echo "  Model file: ${MODEL_FILE}"
    echo "  Output: ${OUTPUT_CSV}"
    
    python scripts/predict_ages.py \
      --model_file "${MODEL_FILE}" \
      --model_type gyro \
      --age_csv data/TIC_cf_data.csv \
      --output_csv "${OUTPUT_CSV}" \
      --ensemble mean

else
    echo "Unknown model type: ${MODEL_TYPE}"
    echo "Usage: ./predict_ages.sh [latent|gyro] [model_version] [output_name]"
    exit 1
fi
