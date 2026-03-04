#!/bin/bash
# K-fold cross-validation for age prediction from latent space + BPRP0
# Each star gets a prediction from a model that never saw it during training
#
# Usage: ./kfold_age_inference.sh [model_path] [version] [pooling_mode] [use_metadata] [use_conv] [conv_type] [require_prot] [h5_path]
# Example: ./kfold_age_inference.sh output/slurm/46126529-lcgen-baseline-flux-only/baseline_flux_only.pt v1 mean false false unet false data/timeseries_x.h5
# Example: ./kfold_age_inference.sh output/simple_rnn/models/baseline_metadata.pt v2 mean true false unet false data/timeseries_x.h5
# Example: ./kfold_age_inference.sh output/simple_rnn/models/baseline_metadata_xchannels.pt v3 mean true true unet false data/timeseries_x.h5
# Example (fair comparison with gyro): ./kfold_age_inference.sh output/simple_rnn/models/baseline.pt v4 mean false false unet true data/timeseries_x.h5
# Example (baseline_v17 model): ./kfold_age_inference.sh output/simple_rnn/models/baseline_v17_real_final.pt v17 multiscale false false unet true data/timeseries.h5
#
# Pooling modes:
#   mean       - Mean pooling over valid timesteps (128 dims for hidden_size=64 bi)
#   multiscale - Multi-scale features: mean, max, std, quartiles, chunks (1536 dims)
#   final      - Final hidden state only (128 dims)

MODEL_PATH=${1:-"output/performance_tests/baseline_long/model.pt"}
VERSION=${2:-"default"}
POOLING_MODE=${3:-"multiscale"}
USE_METADATA=${4:-"false"}
USE_CONV=${5:-"false"}
CONV_TYPE=${6:-"unet"}
REQUIRE_PROT=${7:-"true"}
H5_PATH=${8:-"data/timeseries_x.h5"}

# Select the correct pickle file based on h5 file
# timeseries.h5 -> star_sector_lc_formatted.pickle
# timeseries_x.h5 -> star_sector_lc_formatted_with_extra_channels.pickle
if [[ "${H5_PATH}" == *"timeseries_x"* ]]; then
    PICKLE_PATH="data/star_sector_lc_formatted_with_extra_channels.pickle"
else
    PICKLE_PATH="data/star_sector_lc_formatted.pickle"
fi

OUTPUT_DIR="output/age_predictor/performance_tests/baseline-long-${POOLING_MODE}"

echo "Running k-fold age inference:"
echo "  Model: ${MODEL_PATH}"
echo "  H5 file: ${H5_PATH}"
echo "  Pickle file: ${PICKLE_PATH}"
echo "  Version: ${VERSION}"
echo "  Pooling mode: ${POOLING_MODE}"
echo "  Use metadata: ${USE_METADATA}"
echo "  Use conv channels: ${USE_CONV}"
echo "  Conv encoder type: ${CONV_TYPE}"
echo "  Require Prot: ${REQUIRE_PROT}"
echo "  Output directory: ${OUTPUT_DIR}"

# Build command with conditional flags
CMD="python scripts/kfold_age_inference.py \
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
  --stratify_by_age \
  --n_age_bins 20 \
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

# Add require_prot flag if requested (for fair comparison with gyro baseline)
if [ "${REQUIRE_PROT}" = "true" ]; then
  CMD="${CMD} --require_prot"
fi

# Execute command
eval $CMD
