#!/bin/bash
# K-fold cross-validation for predicting power spectrum and ACF from latent space
# Each sample gets a prediction from a model that never saw it during training
#
# Usage: ./kfold_ps_acf_prediction.sh [model_path] [pooling_mode] [use_metadata] [use_conv] [conv_type] [h5_path]
# Example: ./kfold_ps_acf_prediction.sh output/model.pt multiscale true true unet data/timeseries_x.h5
#
# Pooling modes:
#   mean       - Mean pooling over valid timesteps (128 dims for hidden_size=64 bi)
#   multiscale - Multi-scale features: mean, max, std, quartiles, chunks (1536 dims)
#   final      - Final hidden state only (128 dims)

MODEL_PATH=${1:-"output/performance_tests/baseline_long/model.pt"}
POOLING_MODE=${2:-"multiscale"}
USE_METADATA=${3:-"false"}
USE_CONV=${4:-"false"}
CONV_TYPE=${5:-"unet"}
H5_PATH=${6:-"data/timeseries_x.h5"}

# Select the correct pickle file based on h5 file
if [[ "${H5_PATH}" == *"timeseries_x"* ]]; then
    PICKLE_PATH="data/star_sector_lc_formatted_with_extra_channels.pickle"
else
    PICKLE_PATH="data/star_sector_lc_formatted.pickle"
fi

OUTPUT_DIR="output/ps_acf_predictor/baseline_long_${POOLING_MODE}"

echo "Running k-fold PS/ACF prediction:"
echo "  Model: ${MODEL_PATH}"
echo "  H5 file: ${H5_PATH}"
echo "  Pickle file: ${PICKLE_PATH}"
echo "  Pooling mode: ${POOLING_MODE}"
echo "  Use metadata: ${USE_METADATA}"
echo "  Use conv channels: ${USE_CONV}"
echo "  Conv encoder type: ${CONV_TYPE}"
echo "  Output directory: ${OUTPUT_DIR}"

# Build command with conditional flags
CMD="python scripts/kfold_ps_acf_prediction.py \
  --model_path ${MODEL_PATH} \
  --h5_path ${H5_PATH} \
  --pickle_path ${PICKLE_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --max_length 8192 \
  --pooling_mode ${POOLING_MODE} \
  --target_length 16000 \
  --predict_ps \
  --predict_acf \
  --n_folds 10 \
  --hidden_dims 256 512 1024 \
  --dropout 0.2 \
  --lr 1e-3 \
  --n_epochs 100 \
  --batch_size 64 \
  --seed 42"

# Add metadata flag if requested
if [ "${USE_METADATA}" = "true" ]; then
  CMD="${CMD} --use_metadata"
fi

# Add conv channels flags if requested
if [ "${USE_CONV}" = "true" ]; then
  CMD="${CMD} --use_conv_channels --conv_encoder_type ${CONV_TYPE}"
fi

# Execute command
echo ""
echo "Running command:"
echo "${CMD}"
echo ""
eval $CMD
