#!/bin/bash
# Train a single NLE model on all labeled data (no cross-validation).
# Uses the pre-computed latents cache to avoid re-running the autoencoder.
#
# Usage: ./train_full_nle.sh [version]

VERSION=${1:-"v1"}
OUTPUT_DIR="output/age_predictor/full_nle-${VERSION}"

echo "Training full NLE model:"
echo "  Version:    ${VERSION}"
echo "  Output:     ${OUTPUT_DIR}"

python scripts/train_full_nle.py \
  --load_latents output/latents_cache/cf_final_multiscale.npz \
  --age_csv      data/phot_all.csv \
  --output_dir   "${OUTPUT_DIR}" \
  --star_aggregation latent_max \
  --pca_dim 4 \
  --val_frac 0.1 \
  --lr 1e-3 \
  --lr_decay_rate 0.97 \
  --n_epochs 100 \
  --batch_size 64 \
  --flow_transforms 12 \
  --flow_hidden_dims 64 164 \
  --loga_grid_size 1000 \
  --seed 42
