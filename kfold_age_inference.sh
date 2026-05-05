#!/bin/bash
# K-fold cross-validation for age prediction from latent space + BPRP0.
# Each star gets a prediction from a model that never saw it during training.
#
# Usage:
#   ./kfold_age_inference.sh [model_path] [version] [pooling_mode] [star_aggregation] \
#                            [use_metadata] [use_conv] [conv_type] [require_prot] [h5_path] \
#                            [save_latents] [load_latents]
#
# Latents cache workflow (run model once, compare all aggregation methods cheaply):
#
#   # Step 1 — run model once, save cache (computes both per-sector AND cross-sector latents):
#   ./kfold_age_inference.sh output/performance_tests/baseline_long/model.pt v1 multiscale \
#       cross_sector false false unet true data/timeseries_x.h5 \
#       output/latents_cache/baseline_long_multiscale.npz ""
#
#   # Step 2 — compare aggregation methods by loading from cache (no model required):
#   ./kfold_age_inference.sh "" v1 multiscale latent_mean   ... "" output/latents_cache/baseline_long_multiscale.npz
#   ./kfold_age_inference.sh "" v1 multiscale latent_median ... "" output/latents_cache/baseline_long_multiscale.npz
#   ./kfold_age_inference.sh "" v1 multiscale latent_max    ... "" output/latents_cache/baseline_long_multiscale.npz
#   ./kfold_age_inference.sh "" v1 multiscale cross_sector  ... "" output/latents_cache/baseline_long_multiscale.npz
#   ./kfold_age_inference.sh "" v1 multiscale predict_mean  ... "" output/latents_cache/baseline_long_multiscale.npz
#
# Star aggregation modes (--star_aggregation):
#   none          Legacy: one sample per sector, kfold splits by sector.
#                 WARNING: multi-sector stars can appear in both train and test (leakage).
#   predict_mean  Per-sector predictions; kfold splits by *star*; final metrics average
#                 predictions per star. Per-sector CSV also saved for uncertainty analysis.
#   latent_mean   Sector latents are mean-pooled per star before kfold. No leakage.
#   latent_median Sector latents are median-pooled per star before kfold. No leakage.
#   latent_max    Element-wise max across sector latents per star. No leakage.
#   latent_mean_std  [mean ∥ std] concatenated per star (2× latent dim). No leakage.
#                 Note: single-sector stars get std=0; may introduce confound.
#   cross_sector  Hidden states concatenated across all sectors of a star before
#                 multiscale pooling → one latent per star. No leakage.
#
# Pooling modes (applied per sector before any star aggregation):
#   mean        - Mean pooling over valid timesteps (128 dims for hidden_size=64 bi)
#   multiscale  - Multi-scale temporal features: mean, std, max, min, quartiles (1536 dims)
#   final       - Final hidden state only (128 dims)

MODEL_PATH=${1:-"final_model/parallel_fixed/e110/best_model.pt"}
VERSION=${2:-"default"}
POOLING_MODE=${3:-"multiscale"}
STAR_AGGREGATION=${4:-"predict_mean"}
USE_METADATA=${5:-"true"}
USE_CONV=${6:-"true"}
CONV_TYPE=${7:-"unet"}
REQUIRE_PROT=${8:-"false"}
H5_PATH=${9:-"final_pretrain/timeseries_pretrain.h5"}
SAVE_LATENTS=${10:-""}   # e.g. output/latents_cache/baseline_long_multiscale.npz
LOAD_LATENTS=${11:-"final_model/parallel_fixed/e110/latents_pretrain.npz"}   # e.g. output/latents_cache/baseline_long_multiscale.npz
ENCODER_TYPE=${12:-"mlp"}   # pca, mlp, or linear
USE_MG=${13:-"false"}       # true to include MG_quick as 4th flow context variable
USE_MG_ONLY=${14:-"false"}  # true to swap (BPRP0, BPRP0_err) for (MG, MG_err); mutex with USE_MG

# Optional subset filter — keep only stars where SUBSET_COL is one of SUBSET_VAL.
# Leave SUBSET_COL empty to train on all labeled stars.
# Example: SUBSET_COL="ref"  SUBSET_VAL="ChronoFlow"  SUBSET_CSV="final_pretrain/metadata.csv"
SUBSET_COL=""
SUBSET_VAL=""
SUBSET_CSV=""   # CSV with SUBSET_COL joined on GaiaDR3_ID; defaults to --age_csv if blank

MLP_ENCODER_HIDDEN="128 64"
AUX_LOSS_WEIGHT=1.0
DROPOUT=0.1
VARIANCE_REG_WEIGHT=0.25
TRAINING_STAGES="three_stage"     # joint, two_stage, or three_stage
ENCODER_PRETRAIN_EPOCHS=100
JOINT_FINETUNE_EPOCHS=100
FINETUNE_ENCODER_LR_MULT=0.1  # encoder LR = flow LR * this during stage 3
FINETUNE_FLOW_LR_MULT=0.1       # stage 3 flow LR = peak --lr * this (avoids LR shock from full reset)
TRAIN_FULL="true"               # train a final model on ALL stars after k-fold CV

# H5 files to use for global PCA + normalization (all stars, no age filter)
PCA_H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5"

OUTPUT_DIR="final_model/parallel_fixed/e110/age-inference-${POOLING_MODE}-${STAR_AGGREGATION}"

echo "Running k-fold age inference:"
echo "  Model:            ${MODEL_PATH:-'(from cache)'}"
echo "  H5 file:          ${H5_PATH}"
echo "  Version:          ${VERSION}"
echo "  Pooling mode:     ${POOLING_MODE}"
echo "  Star aggregation: ${STAR_AGGREGATION}"
echo "  Encoder type:     ${ENCODER_TYPE}"
echo "  Training stages:  ${TRAINING_STAGES}"
echo "  Use MG:           ${USE_MG}"
echo "  Use MG only:      ${USE_MG_ONLY}"
echo "  Use metadata:     ${USE_METADATA}"
echo "  Use conv:         ${USE_CONV} (${CONV_TYPE})"
echo "  Require Prot:     ${REQUIRE_PROT}"
echo "  Save latents:     ${SAVE_LATENTS:-'(no)'}"
echo "  Load latents:     ${LOAD_LATENTS:-'(no)'}"
echo "  Output:           ${OUTPUT_DIR}"

# When loading from cache the model_path is unused; omit it to avoid confusion
if [ -n "${LOAD_LATENTS}" ]; then
  CMD="python scripts/kfold_age_inference.py \
    --load_latents ${LOAD_LATENTS}"
else
  CMD="python scripts/kfold_age_inference.py \
    --model_path ${MODEL_PATH} \
    --h5_path ${H5_PATH} \
    --hidden_size 64 \
    --direction bi \
    --mode parallel \
    --use_flow \
    --pooling_mode ${POOLING_MODE} \
    --stratify_by_age \
    --n_age_bins 20"
fi

CMD="${CMD} \
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

if [ -n "${SAVE_LATENTS}" ]; then
  CMD="${CMD} --save_latents ${SAVE_LATENTS}"
fi

if [ "${USE_METADATA}" = "true" ]; then
  CMD="${CMD} --use_metadata"
fi

if [ "${USE_CONV}" = "true" ]; then
  CMD="${CMD} --use_conv_channels --conv_encoder_type ${CONV_TYPE}"
fi

if [ "${REQUIRE_PROT}" = "true" ]; then
  CMD="${CMD} --require_prot"
fi

if [ "${USE_MG}" = "true" ]; then
  CMD="${CMD} --use_mg"
fi

if [ "${USE_MG_ONLY}" = "true" ]; then
  CMD="${CMD} --use_mg_only"
fi

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
