#!/bin/bash
# K-fold age inference on MLP-pooled latents.
# Tests whether a same-budget local-window MLP encoder produces age-informative
# pooled features comparable to the BiDirectionalMinGRU.
#
# Requires extract_mlp_latents.sh to have been run first (produces the cache).
# All parameters hardcoded per project convention.
#
# See docs/plans/2026-05-12_mlp-pooled-age-inference.md.

set -e

LOAD_LATENTS="output/baseline_comparison/mlp_pooled_latents.npz"
OUTPUT_DIR="output/baseline_comparison/age_inference_mlp_pooled"

# Star aggregation: matches the canonical recipe (latent_max).
STAR_AGGREGATION="latent_max"
POOLING_MODE="multiscale"   # nominal; not used in load_latents mode but recorded in output

# Encoder + flow training (mirrors kfold_age_inference.sh).
ENCODER_TYPE="mlp"
MLP_ENCODER_HIDDEN="128 64"
AUX_LOSS_WEIGHT=1.0
DROPOUT=0.1
VARIANCE_REG_WEIGHT=0.25
TRAINING_STAGES="three_stage"
ENCODER_PRETRAIN_EPOCHS=100
JOINT_FINETUNE_EPOCHS=100
FINETUNE_ENCODER_LR_MULT=0.1
FINETUNE_FLOW_LR_MULT=0.1
TRAIN_FULL="true"

PCA_H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5"

# Optional subset filter — mirror the canonical recipe (ChronoFlow subset).
SUBSET_COL="ref"
SUBSET_VAL="ChronoFlow"
SUBSET_CSV="final_pretrain/metadata.csv"

echo "Running MLP-pooled k-fold age inference:"
echo "  Cache:            ${LOAD_LATENTS}"
echo "  Star aggregation: ${STAR_AGGREGATION}"
echo "  Encoder type:     ${ENCODER_TYPE}"
echo "  Training stages:  ${TRAINING_STAGES}"
echo "  Output:           ${OUTPUT_DIR}"

CMD="python scripts/kfold_age_inference.py \
  --load_latents ${LOAD_LATENTS} \
  --age_csv final_pretrain/all_ages.csv \
  --pca_h5_paths ${PCA_H5_PATHS} \
  --output_dir ${OUTPUT_DIR} \
  --star_aggregation ${STAR_AGGREGATION} \
  --n_folds 10 \
  --encoder_type ${ENCODER_TYPE} \
  --pca_dim 4 \
  --mlp_encoder_hidden ${MLP_ENCODER_HIDDEN} \
  --lr 1e-3 \
  --lr_decay_rate 0.97 \
  --n_epochs 300 \
  --batch_size 64 \
  --flow_transforms 6 \
  --flow_hidden_dims 64 64 \
  --loga_grid_size 1000 \
  --seed 42 \
  --aux_loss_weight ${AUX_LOSS_WEIGHT} \
  --dropout ${DROPOUT} \
  --variance_reg_weight ${VARIANCE_REG_WEIGHT} \
  --training_stages ${TRAINING_STAGES} \
  --encoder_pretrain_epochs ${ENCODER_PRETRAIN_EPOCHS} \
  --joint_finetune_epochs ${JOINT_FINETUNE_EPOCHS} \
  --finetune_encoder_lr_mult ${FINETUNE_ENCODER_LR_MULT} \
  --finetune_flow_lr_mult ${FINETUNE_FLOW_LR_MULT}"

if [ "${TRAIN_FULL}" = "true" ]; then
  CMD="${CMD} --train_full"
fi

if [ -n "${SUBSET_COL}" ] && [ -n "${SUBSET_VAL}" ]; then
  CMD="${CMD} --subset_col ${SUBSET_COL} --subset_val ${SUBSET_VAL}"
  if [ -n "${SUBSET_CSV}" ]; then
    CMD="${CMD} --subset_csv ${SUBSET_CSV}"
  fi
fi

eval $CMD
