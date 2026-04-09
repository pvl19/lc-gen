#!/bin/bash
# Run k-fold age inference with all pooling modes for comparison
#
# Usage: ./run_all_pooling_modes.sh [version]
# Example: ./run_all_pooling_modes.sh v1
#
# This will create:
#   output/age_predictor/kfold_v1_mean/
#   output/age_predictor/kfold_v1_multiscale/
#   output/age_predictor/kfold_v1_final/

VERSION=${1:-"v1"}

echo "=============================================="
echo "Running all pooling modes with version: ${VERSION}"
echo "=============================================="

# Run mean pooling
echo ""
echo ">>> Running MEAN pooling..."
../../kfold_age_inference.sh "${VERSION}" mean

# Run multiscale pooling
echo ""
echo ">>> Running MULTISCALE pooling..."
../../kfold_age_inference.sh "${VERSION}" multiscale

# Run final pooling
echo ""
echo ">>> Running FINAL pooling..."
../../kfold_age_inference.sh "${VERSION}" final

echo ""
echo "=============================================="
echo "All pooling modes complete!"
echo ""
echo "Results saved to:"
echo "  output/age_predictor/kfold_${VERSION}_mean/"
echo "  output/age_predictor/kfold_${VERSION}_multiscale/"
echo "  output/age_predictor/kfold_${VERSION}_final/"
echo ""
echo "To compare results, run:"
echo "  ../../bin/plot/compare_pooling_modes.sh ${VERSION}"
echo "=============================================="
