#!/bin/bash
# Combined pretrain + host k-fold age inference.
#
# Trains the age flow on (all non-host pretrain stars) ∪ ((K-1)/K of hosts) per
# fold; validates on the held-out host fold. Reports host-only metrics. After
# k-fold, optionally trains a deployment model on (hosts ∪ non-hosts).
#
# All parameters are hardcoded below. Latents caches for each source can be
# reused independently via SAVE_LATENTS_PRETRAIN / LOAD_LATENTS_PRETRAIN and
# SAVE_LATENTS_HOSTS / LOAD_LATENTS_HOSTS. Either side may be loaded from
# cache while the other is freshly extracted.
#
# Plan: docs/plans/2026-04-29_combined-pretrain-host-kfold.md

MODEL_PATH="final_model/parallel_fixed/e60/model.pt"
VERSION="default"
POOLING_MODE="multiscale"
STAR_AGGREGATION="latent_max"
USE_METADATA="true"
USE_CONV="true"
CONV_TYPE="unet"
REQUIRE_PROT="false"

# Pretrain (non-host) source
H5_PATH="final_pretrain/timeseries_pretrain.h5"
AGE_CSV="final_pretrain/all_ages.csv"

# Host source (NASA Exoplanet Archive)
HOST_H5_PATH="final_pretrain/timeseries_exop_hosts.h5"
HOST_AGE_CSV="exop_hosts/archive_ages_default.csv"
HOST_METADATA_CSV="final_pretrain/host_all_metadata.csv"

# Independent latent caches per source
SAVE_LATENTS_PRETRAIN=""
SAVE_LATENTS_HOSTS="final_model/parallel_fixed/e60/combined_age_inference/latents_hosts.npz"
LOAD_LATENTS_PRETRAIN="final_model/parallel_fixed/e60/latents.npz"    # set to skip pretrain extraction
LOAD_LATENTS_HOSTS=""       # set to skip host extraction

ENCODER_TYPE="mlp"
USE_MG="false"
USE_MG_ONLY="false"     # true to swap (BPRP0, BPRP0_err) for (MG, MG_err); mutex with USE_MG
MLP_ENCODER_HIDDEN="128 64"
AUX_LOSS_WEIGHT=1.0
DROPOUT=0.1
VARIANCE_REG_WEIGHT=0.25
TRAINING_STAGES="three_stage"
ENCODER_PRETRAIN_EPOCHS=100
JOINT_FINETUNE_EPOCHS=100
FINETUNE_ENCODER_LR_MULT=0.001
FINETUNE_FLOW_LR_MULT=0.1
TRAIN_FULL="true"

# H5 files used to fit global PCA + latent normalization (no age filter).
# Both files are also the data sources, so this just guarantees the global
# stats cover both populations when extraction runs fresh.
PCA_H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5"

OUTPUT_DIR="final_model/parallel_fixed/e60/combined_age_inference/3stage-10f-combined-${POOLING_MODE}-${STAR_AGGREGATION}"

echo "Running combined k-fold age inference:"
echo "  Model:                 ${MODEL_PATH}"
echo "  Pretrain H5:           ${H5_PATH}"
echo "  Pretrain age CSV:      ${AGE_CSV}"
echo "  Host H5:               ${HOST_H5_PATH}"
echo "  Host age CSV:          ${HOST_AGE_CSV}"
echo "  Host metadata CSV:     ${HOST_METADATA_CSV}"
echo "  Pooling mode:          ${POOLING_MODE}"
echo "  Star aggregation:      ${STAR_AGGREGATION}"
echo "  Encoder type:          ${ENCODER_TYPE}"
echo "  Training stages:       ${TRAINING_STAGES}"
echo "  Use MG:                ${USE_MG}"
echo "  Use MG only:           ${USE_MG_ONLY}"
echo "  Use metadata:          ${USE_METADATA}"
echo "  Use conv:              ${USE_CONV} (${CONV_TYPE})"
echo "  Save latents pretrain: ${SAVE_LATENTS_PRETRAIN:-'(no)'}"
echo "  Load latents pretrain: ${LOAD_LATENTS_PRETRAIN:-'(no)'}"
echo "  Save latents hosts:    ${SAVE_LATENTS_HOSTS:-'(no)'}"
echo "  Load latents hosts:    ${LOAD_LATENTS_HOSTS:-'(no)'}"
echo "  Output:                ${OUTPUT_DIR}"

CMD="python scripts/kfold_age_inference.py \
  --combined_kfold \
  --model_path ${MODEL_PATH} \
  --h5_path ${H5_PATH} \
  --age_csv ${AGE_CSV} \
  --host_h5_path ${HOST_H5_PATH} \
  --host_age_csv ${HOST_AGE_CSV} \
  --host_metadata_csv ${HOST_METADATA_CSV} \
  --pca_h5_paths ${PCA_H5_PATHS} \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --pooling_mode ${POOLING_MODE} \
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

if [ -n "${SAVE_LATENTS_PRETRAIN}" ]; then
  CMD="${CMD} --save_latents_pretrain ${SAVE_LATENTS_PRETRAIN}"
fi
if [ -n "${LOAD_LATENTS_PRETRAIN}" ]; then
  CMD="${CMD} --load_latents_pretrain ${LOAD_LATENTS_PRETRAIN}"
fi
if [ -n "${SAVE_LATENTS_HOSTS}" ]; then
  CMD="${CMD} --save_latents_hosts ${SAVE_LATENTS_HOSTS}"
fi
if [ -n "${LOAD_LATENTS_HOSTS}" ]; then
  CMD="${CMD} --load_latents_hosts ${LOAD_LATENTS_HOSTS}"
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

eval $CMD
