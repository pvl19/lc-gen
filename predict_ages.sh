#!/bin/bash
# Run age inference on unlabeled stars using a trained full NLE model.
#
# First run extracts latents from the autoencoder and caches them.
# Subsequent runs can skip the autoencoder by passing a cached latents file.
#
# Usage:
#   ./predict_ages.sh [nle_version] [h5_path] [load_latents]
#
# Examples:
#   # First run — extract latents from autoencoder:
#   ./predict_ages.sh v1 exop_hosts/lc_data.h5 ""
#
#   # Subsequent runs — load from cache:
#   ./predict_ages.sh v1 "" output/exop_host_ages-v1/latents.npz

NLE_VERSION=${1:-"v1"}
H5_PATH=${2:-"exop_hosts/lc_data.h5"}
LOAD_LATENTS=${3:-""}

NLE_MODEL="output/age_predictor/full_nle-${NLE_VERSION}/full_nle_model.pt"
AE_MODEL="output/performance_tests/cf_final/model.pt"
OUTPUT_DIR="exop_hosts/exop_host_ages-${NLE_VERSION}"
SAVE_LATENTS="${OUTPUT_DIR}/latents.npz"

echo "Running age inference:"
echo "  NLE model:    ${NLE_MODEL}"
echo "  AE model:     ${AE_MODEL}"
echo "  H5 path:      ${H5_PATH:-'(from cache)'}"
echo "  Load latents: ${LOAD_LATENTS:-'(no — will extract)'}"
echo "  Output:       ${OUTPUT_DIR}"

if [ -n "${LOAD_LATENTS}" ]; then
  python scripts/predict_ages.py \
    --nle_model    "${NLE_MODEL}" \
    --load_latents "${LOAD_LATENTS}" \
    --output_dir   "${OUTPUT_DIR}" \
    --loga_grid_size 1000 \
    --batch_size 256
else
  python scripts/predict_ages.py \
    --nle_model    "${NLE_MODEL}" \
    --ae_model     "${AE_MODEL}" \
    --h5_path      "${H5_PATH}" \
    --output_dir   "${OUTPUT_DIR}" \
    --save_latents "${SAVE_LATENTS}" \
    --hidden_size 64 \
    --direction bi \
    --mode parallel \
    --max_length 8192 \
    --pooling_mode multiscale \
    --loga_grid_size 1000 \
    --batch_size 64 \
    --nle_batch_size 256
fi
