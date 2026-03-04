#!/bin/bash
# K-fold cross-validation for Prot (rotation period) prediction from latent space + BPRP0
# Each star gets a prediction from a model that never saw it during training
#
# Usage: ./kfold_prot_inference.sh [model_path] [version] [pooling_mode] [use_metadata] [use_conv] [conv_type] [h5_path]
# Example: ./kfold_prot_inference.sh output/slurm/46126529-lcgen-baseline-flux-only/baseline_flux_only.pt v1 mean false false unet data/timeseries_x.h5
# Example: ./kfold_prot_inference.sh output/simple_rnn/models/baseline_metadata.pt v2 mean true false unet data/timeseries_x.h5
# Example: ./kfold_prot_inference.sh output/simple_rnn/models/baseline_metadata_xchannels.pt v3 mean true true unet data/timeseries_x.h5
#
# Pooling modes:
#   mean       - Mean pooling over valid timesteps (128 dims for hidden_size=64 bi)
#   multiscale - Multi-scale features: mean, max, std, quartiles, chunks (1536 dims)
#   final      - Final hidden state only (128 dims)

MODEL_PATH=${1:-"output/performance_tests/long_curve/epoch30/model.pt"}
VERSION=${2:-"default"}
POOLING_MODE=${3:-"multiscale"}
USE_METADATA=${4:-"true"}
USE_CONV=${5:-"true"}
CONV_TYPE=${6:-"unet"}
H5_PATH=${7:-"data/timeseries_x.h5"}

# Select the correct pickle file based on h5 file
# timeseries.h5 -> star_sector_lc_formatted.pickle
# timeseries_x.h5 -> star_sector_lc_formatted_with_extra_channels.pickle
if [[ "${H5_PATH}" == *"timeseries_x"* ]]; then
    PICKLE_PATH="data/star_sector_lc_formatted_with_extra_channels.pickle"
else
    PICKLE_PATH="data/star_sector_lc_formatted.pickle"
fi

OUTPUT_DIR="output/prot_predictor/long_curve-${POOLING_MODE}"

echo "Running k-fold Prot inference:"
echo "  Model: ${MODEL_PATH}"
echo "  H5 file: ${H5_PATH}"
echo "  Pickle file: ${PICKLE_PATH}"
echo "  Version: ${VERSION}"
echo "  Pooling mode: ${POOLING_MODE}"
echo "  Use metadata: ${USE_METADATA}"
echo "  Use conv channels: ${USE_CONV}"
echo "  Conv encoder type: ${CONV_TYPE}"
echo "  Output directory: ${OUTPUT_DIR}"

# Build command with conditional flags
CMD="python scripts/kfold_prot_inference.py \
  --model_path ${MODEL_PATH} \
  --h5_path ${H5_PATH} \
  --pickle_path ${PICKLE_PATH} \
  --age_csv data/TIC_cf_data.csv \
  --output_dir ${OUTPUT_DIR} \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --max_length 8192 \
  --pooling_mode ${POOLING_MODE} \
  --n_folds 10 \
  --hidden_dims 128 64 32 \
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
eval $CMD
