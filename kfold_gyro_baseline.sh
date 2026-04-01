#!/bin/bash
# Gyrochronology NLE baseline: p(log10(Prot) | log10_age, BPRP0, log10(BPRP0_err))
# Mirrors kfold_age_inference.sh but uses Prot as the 1D likelihood variable.
#
# Usage: ./kfold_gyro_baseline.sh [version]

VERSION=${1:-"v1"}
LOAD_LATENTS=${2:-"output/latents_cache/cf_final_multiscale.npz"}
OUTPUT_DIR="output/age_predictor_tests/gyro-nle-${VERSION}"

echo "Running gyro NLE baseline:"
echo "  Version:      ${VERSION}"
echo "  Load latents: ${LOAD_LATENTS:-'(no filter)'}"
echo "  Output:       ${OUTPUT_DIR}"

python scripts/kfold_gyro_baseline.py \
  --age_csv data/phot_all.csv \
  --output_dir "${OUTPUT_DIR}" \
  --flow_transforms 8 \
  --flow_hidden_dims 64 64 \
  --lr 1e-3 \
  --lr_decay_rate 0.97 \
  --n_epochs 100 \
  --batch_size 64 \
  --n_folds 5 \
  --loga_grid_size 1000 \
  --seed 42 \
  ${LOAD_LATENTS:+--load_latents "${LOAD_LATENTS}"}
