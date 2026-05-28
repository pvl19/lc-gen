#!/bin/bash
# K-fold NLE / NPE age inference for exoplanet hosts on the FINAL sendit/e50 latents.
# (Canonical configurable host launcher — edit the variables below for any test.)
#
# Mirrors the PCA4 pretrain workflow (kfold_pca_dimsweep.sh / kfold_loso.sh):
#   - PCA encoder, dim 4
#   - Global PCA basis from the shared cache (fit on pretrain+hosts+thickdisk
#     at dim 16, truncated to 4) — bit-identical to the pretrain PCA4 runs
#   - latent_max per-star aggregation
#   - Simple stratified 10-fold by star (single per-star prediction)
#
# Main toggles (all explained inline below):
#   PREDICTION_MODE       nle (Bayes-grid posterior) | npe (direct age flow)
#   NPE_STANDARDIZE_TARGET z-score the NPE age target (lifts the ±5-spline cap)
#   AGE_SPACE             gyr | log10_myr | passthrough
#   EIV                   errors-in-variables on noisy literature ages
#   K_AGE_SAMPLES         Gaussian age-error propagation (mutually excl. w/ EIV)
#
# All parameters hardcoded per project convention. Calls scripts/kfold_nle_age_inference_hosts.py.

LAT_HOSTS=final_model/sendit/e50/metaAll/latents_hosts.npz
LAT_PRETRAIN=final_model/sendit/e50/metaAll/latents_pretrain.npz
LAT_THICK=final_model/sendit/e50/metaAll/latents_thickdisk.npz

HOST_AGE_CSV=exop_hosts/archive_ages_normalized.csv
HOST_METADATA_CSV=final_pretrain/host_all_metadata.csv

# Age target / sampling configuration. Flip these to switch variants:
#   AGE_SPACE=log10_myr  HOST_AGE_COL=st_age      HOST_AGE_ERR_COL=st_ageerr      (default; matches pretrain log-age space)
#   AGE_SPACE=gyr        HOST_AGE_COL=st_age      HOST_AGE_ERR_COL=st_ageerr      (flow learns Gyr directly)
#   AGE_SPACE=passthrough HOST_AGE_COL=st_age_norm HOST_AGE_ERR_COL=st_ageerr_norm (Bouma-normalized; column fed as-is)
# K_AGE_SAMPLES=1 disables Gaussian age sampling; K_AGE_SAMPLES=10 propagates the per-star σ
# from HOST_AGE_ERR_COL by drawing 10 samples in CSV units, transforming each per AGE_SPACE,
# and averaging the per-star NLL over the K draws.
#
# EIV=true enables errors-in-variables (LatentNN-style): each star gets a learnable age
# latent jointly optimized with the flow, plus a Gaussian-prior regularizer
# ((y_obs−y_latent)/σ)² that anchors it to the literature value. Mutually exclusive with
# K_AGE_SAMPLES>1 (forced to 1 below when EIV=true). Currently PCA + joint stages only.
AGE_SPACE=gyr
HOST_AGE_COL=st_age
HOST_AGE_ERR_COL=st_ageerr
K_AGE_SAMPLES=1
EIV=true
EIV_SIGMA_FLOOR_FRAC=0.05

if [ "${EIV}" = "true" ] && [ "${K_AGE_SAMPLES}" -gt 1 ]; then
  echo "EIV=true is mutually exclusive with K_AGE_SAMPLES>1; forcing K_AGE_SAMPLES=1."
  K_AGE_SAMPLES=1
fi

# Shared global PCA artifact (created by kfold_pca_dimsweep.sh on its first
# pretrain run; reused as-is here if it exists, else built from the same
# 3-cache pool so this script is self-bootstrapping).
AGE_ROOT=final_model/sendit/e50/age_inference
PCA_CACHE=${AGE_ROOT}/shared/global_pca_d16.npz
PCA_POOL_OR_CACHE="--pca_latent_pool ${LAT_PRETRAIN} ${LAT_HOSTS} ${LAT_THICK} \
  --pca_cache ${PCA_CACHE} --pca_cache_max_dim 16"

ENCODER_TYPE=pca
BOTTLENECK_DIM=4
TRAINING_STAGES=joint            # no learnable encoder to pre-train when PCA

# Prediction mode (the toggle):
#   nle — flow models p(z_pca | age, colour, colour_err); age posterior recovered
#         by Bayes-inverting a likelihood grid over the age context (the default).
#   npe — flow conditioned on the 4-dim PCA + colour + colour_err DIRECTLY, with
#         age as the 1D flow output. The age prediction is read straight off the
#         flow's own density (no grid inversion). Same PCA features, same
#         predictions.csv format; only the conditioning direction flips.
PREDICTION_MODE=npe

# NPE target standardization (NPE only; no-op for NLE). The NSF spline has a hard
# ±5 support, so an NPE flow modelling age DIRECTLY caps at ~5 Gyr unless the
# target is z-scored. true = z-score by ONE global (loc, scale) over all stars,
# shared across folds + the full model and baked into the saved model buffers;
# predictions stay in AGE_SPACE units, no manual re-standardization on reload.
# Safe to leave on; only set false if AGE_SPACE already fits ±5 (e.g. log10_myr).
NPE_STANDARDIZE_TARGET=true

STAR_AGGREGATION=latent_max
USE_MG=false

# Note: the static-outlier mixture (P_OUTLIER, P_CLUSTER_MEM) is HARDCODED OFF
# inside scripts/kfold_nle_age_inference_hosts.py — hosts have no cluster-
# membership concept, so the training NLL is the pure flow log-prob.

N_FOLDS=10
SEED=42

LR=1e-3
LR_DECAY_RATE=0.97
WEIGHT_DECAY=1e-4
N_EPOCHS=100                     # joint-only training; matches kfold_loso.sh / kfold_pca_dimsweep.sh
BATCH_SIZE=64

FLOW_TRANSFORMS=6
FLOW_HIDDEN_DIMS="64 64"
LOGA_GRID_SIZE=1000

TRAIN_FULL=true

OUTPUT_DIR=${AGE_ROOT}/hosts/pca${BOTTLENECK_DIM}_${STAR_AGGREGATION}_${AGE_SPACE}_K${K_AGE_SAMPLES}
if [ "${PREDICTION_MODE}" != "nle" ]; then OUTPUT_DIR="${OUTPUT_DIR}_${PREDICTION_MODE}"; fi
if [ "${EIV}" = "true" ]; then OUTPUT_DIR="${OUTPUT_DIR}_EIV"; fi

echo "Running k-fold $(echo "${PREDICTION_MODE}" | tr '[:lower:]' '[:upper:]') age inference (hosts, sendit/e50 PCA${BOTTLENECK_DIM}):"
echo "  Latents:          ${LAT_HOSTS}"
echo "  Host age CSV:     ${HOST_AGE_CSV}  col=${HOST_AGE_COL}  err_col=${HOST_AGE_ERR_COL}"
echo "  Age space:        ${AGE_SPACE}  (K=${K_AGE_SAMPLES} Gaussian samples per star)"
echo "  Prediction mode:  ${PREDICTION_MODE}  (npe target standardize=${NPE_STANDARDIZE_TARGET}, EIV=${EIV})"
echo "  Host metadata:    ${HOST_METADATA_CSV}"
echo "  PCA cache:        ${PCA_CACHE}"
echo "  Star aggregation: ${STAR_AGGREGATION}"
echo "  Encoder:          ${ENCODER_TYPE}  dim=${BOTTLENECK_DIM}"
echo "  Folds / seed:     ${N_FOLDS} / ${SEED}"
echo "  Output:           ${OUTPUT_DIR}"

CMD="python scripts/kfold_nle_age_inference_hosts.py \
  --load_latents ${LAT_HOSTS} \
  --host_age_csv ${HOST_AGE_CSV} \
  --host_age_col ${HOST_AGE_COL} \
  --age_space ${AGE_SPACE} \
  --k_age_samples ${K_AGE_SAMPLES} \
  --host_metadata_csv ${HOST_METADATA_CSV} \
  --output_dir ${OUTPUT_DIR} \
  --star_aggregation ${STAR_AGGREGATION} \
  --prediction_mode ${PREDICTION_MODE} \
  --n_folds ${N_FOLDS} \
  --seed ${SEED} \
  --encoder_type ${ENCODER_TYPE} \
  --bottleneck_dim ${BOTTLENECK_DIM} \
  --training_stages ${TRAINING_STAGES} \
  --lr ${LR} \
  --lr_decay_rate ${LR_DECAY_RATE} \
  --weight_decay ${WEIGHT_DECAY} \
  --n_epochs ${N_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --flow_transforms ${FLOW_TRANSFORMS} \
  --flow_hidden_dims ${FLOW_HIDDEN_DIMS} \
  --loga_grid_size ${LOGA_GRID_SIZE} \
  ${PCA_POOL_OR_CACHE}"

if [ "${USE_MG}"     = "true" ]; then CMD="${CMD} --use_mg"; fi
if [ "${TRAIN_FULL}" = "true" ]; then CMD="${CMD} --train_full"; fi
if [ "${K_AGE_SAMPLES}" -gt 1 ] || [ "${EIV}" = "true" ]; then
  CMD="${CMD} --host_age_err_col ${HOST_AGE_ERR_COL}"
fi
if [ "${EIV}" = "true" ]; then
  CMD="${CMD} --eiv --eiv_sigma_floor_frac ${EIV_SIGMA_FLOOR_FRAC}"
fi
if [ "${NPE_STANDARDIZE_TARGET}" = "true" ]; then
  CMD="${CMD} --npe_standardize_target"
else
  CMD="${CMD} --no-npe_standardize_target"
fi

eval $CMD
