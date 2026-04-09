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

MODEL_PATH=${1:-"final_model/final_parallel_e10/model.pt"}
VERSION=${2:-"default"}
POOLING_MODE=${3:-"multiscale"}
STAR_AGGREGATION=${4:-"latent_max"}
USE_METADATA=${5:-"true"}
USE_CONV=${6:-"false"}
CONV_TYPE=${7:-"unet"}
REQUIRE_PROT=${8:-"false"}
H5_PATH=${9:-"final_pretrain/timeseries_pretrain.h5"}
SAVE_LATENTS=${10:-""}   # e.g. output/latents_cache/baseline_long_multiscale.npz
LOAD_LATENTS=${11:-"final_model/final_parallel_e10/latents.npz"}   # e.g. output/latents_cache/baseline_long_multiscale.npz
ENCODER_TYPE=${12:-"pca"}   # pca or mlp
USE_MG=${13:-"false"}       # true to include log10(MG_quick) as 4th flow context variable
MLP_ENCODER_HIDDEN="128 64"
AUX_LOSS_WEIGHT=1.0
DROPOUT=0.1
VARIANCE_REG_WEIGHT=0.25

# H5 files to use for global PCA + normalization (all stars, no age filter)
PCA_H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5"

OUTPUT_DIR="final_model/final_parallel_e10/nf-pca64-${POOLING_MODE}-${STAR_AGGREGATION}-${ENCODER_TYPE}"

echo "Running k-fold age inference:"
echo "  Model:            ${MODEL_PATH:-'(from cache)'}"
echo "  H5 file:          ${H5_PATH}"
echo "  Version:          ${VERSION}"
echo "  Pooling mode:     ${POOLING_MODE}"
echo "  Star aggregation: ${STAR_AGGREGATION}"
echo "  Encoder type:     ${ENCODER_TYPE}"
echo "  Use MG:           ${USE_MG}"
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
  --n_folds 5 \
  --encoder_type ${ENCODER_TYPE} \
  --pca_dim 64 \
  --mlp_encoder_hidden ${MLP_ENCODER_HIDDEN} \
  --lr 1e-3 \
  --lr_decay_rate 0.97 \
  --n_epochs 200 \
  --batch_size 64 \
  --flow_transforms 6 \
  --flow_hidden_dims 64 64 \
  --loga_grid_size 1000 \
  --seed 42 \
  --aux_loss_weight ${AUX_LOSS_WEIGHT} \
  --dropout ${DROPOUT} \
  --variance_reg_weight ${VARIANCE_REG_WEIGHT}"

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

eval $CMD
