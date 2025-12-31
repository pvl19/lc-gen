#!/bin/bash
# Compare age prediction results across different pooling modes
#
# Usage: ./compare_pooling_modes.sh [version]
# Example: ./compare_pooling_modes.sh v1
#
# Prerequisite: Run ./run_all_pooling_modes.sh v1 first

VERSION=${1:-"v1"}
OUTPUT_DIR="output/age_predictor/pooling_comparison_${VERSION}"

echo "Comparing pooling modes for version: ${VERSION}"
echo "Output directory: ${OUTPUT_DIR}"

python scripts/compare_pooling_modes.py \
  --version "${VERSION}" \
  --base_dir output/age_predictor \
  --output_dir "${OUTPUT_DIR}" \
  --pooling_modes mean multiscale final
