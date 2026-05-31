#!/bin/bash
# K-fold NLE / NPE age inference for exoplanet hosts on the FINAL sendit/e100 latents.
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
#   BALANCE_AGE           flatten the age prior (de-piles NPE's data-mode pull)
#   AGE_SPACE             gyr | log10_myr | passthrough
#   EIV                   errors-in-variables on noisy literature ages
#   K_AGE_SAMPLES         Gaussian age-error propagation (mutually excl. w/ EIV)
#
# All parameters hardcoded per project convention. Calls scripts/kfold_nle_age_inference_hosts.py.

LAT_HOSTS=final_model/sendit/e100/latents_hosts_merged.npz
LAT_PRETRAIN=final_model/sendit/e100/latents_pretrain_merged.npz
LAT_THICK=final_model/sendit/e100/latents_thickdisk_merged.npz

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
AGE_ROOT=final_model/sendit/e100/age_inference
PCA_CACHE=${AGE_ROOT}/shared/global_pca_d16.npz
PCA_POOL_OR_CACHE="--pca_latent_pool ${LAT_PRETRAIN} ${LAT_HOSTS} ${LAT_THICK} \
  --pca_cache ${PCA_CACHE} --pca_cache_max_dim 16"

# Encoder over the 1536-dim latent:
#   pca         — fixed PCA projection (uses the shared PCA cache below).
#   mlp / linear — a LEARNED encoder that compresses the FULL 1536-dim latent to
#                  BOTTLENECK_DIM, trained jointly with the flow. An auxiliary age
#                  head + variance reg keep the bottleneck from collapsing. The PCA
#                  cache is ignored. EIV works here too, but joint-training only
#                  (forced below); use K_AGE_SAMPLES>1 as the alternative.
ENCODER_TYPE=mlp
BOTTLENECK_DIM=4
TRAINING_STAGES=joint            # joint | two_stage | three_stage (mlp/linear only)

# Learned-encoder (mlp/linear) hyperparameters — ignored when ENCODER_TYPE=pca.
#   DROPOUT       — dropout after each hidden layer.
#   INPUT_DROPOUT — input feature masking on the raw 1536-d latent (a stronger
#                   regularizer for the wide input; 0.0 = off). Bump for stability
#                   if the train/val NLL gap is wide.
MLP_ENCODER_HIDDEN="256 128"
AUX_LOSS_WEIGHT=1.0
DROPOUT=0.1
INPUT_DROPOUT=0.1
VARIANCE_REG_WEIGHT=0.25

# EIV works for pca AND mlp/linear (the flow consumes the learnable latent age;
# the mlp aux head keeps targeting y_obs), but ONLY with joint training — staged
# schedules need per-stage y_obs/y_latent routing that isn't wired. Force joint
# when EIV is on so the script stays runnable.
if [ "${EIV}" = "true" ] && [ "${TRAINING_STAGES}" != "joint" ]; then
  echo "EIV requires TRAINING_STAGES=joint; forcing joint (was ${TRAINING_STAGES})."
  TRAINING_STAGES=joint
fi

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

# Age-marginal balancing (mainly for NPE). NPE learns p(age|z) ∝ p(z|age)·p(age),
# so a peaked training age distribution (hosts pile up ~4 Gyr) drags weakly-
# conditioned predictions toward that mode. true = WeightedRandomSampler flattens
# the age histogram so the flow sees a ~flat prior → behaves like a likelihood
# (flat-prior) estimator, de-piling the mode. BALANCE_AGE_TEMP sets the strength:
# 1.0 fully flattens; <1 softens (less oversampling of the sparse old tail →
# lower variance); 0 ≡ off. Output dir gets a _balageT<temp> suffix so balanced
# and unbalanced runs don't clobber each other.
BALANCE_AGE=true
N_BALANCE_AGE_BINS=10
BALANCE_AGE_TEMP=1.0

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

# Draw K samples per star from the held-out grid posterior (inverse CDF, seeded
# by SEED) and save as `posterior_samples` in heldout_posteriors.npz so the
# pooled-residual violin script (scripts/plot_host_residual_violins.py) doesn't
# have to redo the sampling. 0 = skip (back-compat). 200 is plenty: a typical
# 100-star bin pools to 20k samples per violin, far above what KDE needs.
N_POSTERIOR_SAMPLES=200

# Noise-baseline mode: when true, substitute LAT_HOSTS for a sidecar npz where
# `latent_vectors` is replaced by per-feature mean/std-matched Gaussian noise
# (all identifiers / ages / metadata preserved). The whole pipeline — encoder,
# flow, EIV, balance, posterior sampling — runs unchanged on the noise; the
# resulting metrics + heldout_posteriors.npz become the "is our latent better
# than random features of the same scale" control. Output is routed to a
# parallel `_NOISE` subdir so it doesn't clobber the real run. The sidecar is
# generated on first use (seeded by SEED for reproducibility).
NOISE_BASELINE=false

TRAIN_FULL=true

# Encoder-specific CLI args: PCA feeds the shared cache; mlp/linear feed the
# learned-encoder hyperparameters (and ignore the cache entirely).
if [ "${ENCODER_TYPE}" = "pca" ]; then
  ENCODER_EXTRA_ARGS="${PCA_POOL_OR_CACHE}"
else
  ENCODER_EXTRA_ARGS="--mlp_encoder_hidden ${MLP_ENCODER_HIDDEN} \
    --aux_loss_weight ${AUX_LOSS_WEIGHT} --dropout ${DROPOUT} \
    --input_dropout ${INPUT_DROPOUT} --variance_reg_weight ${VARIANCE_REG_WEIGHT}"
fi

# Noise-baseline path swap: generate the sidecar if missing, then redirect the
# host latents input. Done BEFORE OUTPUT_DIR is built so the _NOISE suffix gets
# tacked on. Keeps the original LAT_HOSTS path so the real bank stays the source
# of truth for the per-feature mean/std match.
if [ "${NOISE_BASELINE}" = "true" ]; then
  REAL_LAT_HOSTS=${LAT_HOSTS}
  NOISE_LAT_HOSTS="${REAL_LAT_HOSTS%.npz}_noise_seed${SEED}.npz"
  if [ ! -f "${NOISE_LAT_HOSTS}" ]; then
    echo "Generating noise-baseline latents -> ${NOISE_LAT_HOSTS}"
    python scripts/make_noise_baseline_latents.py \
      --in_npz "${REAL_LAT_HOSTS}" \
      --out_npz "${NOISE_LAT_HOSTS}" \
      --seed ${SEED} --match per_feature \
      || { echo "  !! NOISE BASELINE GENERATION FAILED"; exit 1; }
  else
    echo "Reusing existing noise-baseline latents: ${NOISE_LAT_HOSTS}"
  fi
  LAT_HOSTS=${NOISE_LAT_HOSTS}
fi

OUTPUT_DIR=${AGE_ROOT}/hosts/${ENCODER_TYPE}${BOTTLENECK_DIM}_${STAR_AGGREGATION}_${AGE_SPACE}_K${K_AGE_SAMPLES}
if [ "${PREDICTION_MODE}" != "nle" ]; then OUTPUT_DIR="${OUTPUT_DIR}_${PREDICTION_MODE}"; fi
if [ "${EIV}" = "true" ]; then OUTPUT_DIR="${OUTPUT_DIR}_EIV"; fi
if [ "${BALANCE_AGE}" = "true" ]; then OUTPUT_DIR="${OUTPUT_DIR}_balageT${BALANCE_AGE_TEMP}"; fi
if [ "${ENCODER_TYPE}" != "pca" ] && [ "${INPUT_DROPOUT}" != "0" ] && [ "${INPUT_DROPOUT}" != "0.0" ]; then
  OUTPUT_DIR="${OUTPUT_DIR}_indrop${INPUT_DROPOUT}"
fi
if [ "${USE_MG}" = "true" ]; then OUTPUT_DIR="${OUTPUT_DIR}_MG"; fi
if [ "${NOISE_BASELINE}" = "true" ]; then OUTPUT_DIR="${OUTPUT_DIR}_NOISE"; fi

echo "Running k-fold $(echo "${PREDICTION_MODE}" | tr '[:lower:]' '[:upper:]') age inference (hosts, sendit/e100 ${ENCODER_TYPE}${BOTTLENECK_DIM}):"
echo "  Latents:          ${LAT_HOSTS}"
echo "  Host age CSV:     ${HOST_AGE_CSV}  col=${HOST_AGE_COL}  err_col=${HOST_AGE_ERR_COL}"
echo "  Age space:        ${AGE_SPACE}  (K=${K_AGE_SAMPLES} Gaussian samples per star)"
echo "  Prediction mode:  ${PREDICTION_MODE}  (npe target standardize=${NPE_STANDARDIZE_TARGET}, EIV=${EIV})"
echo "  Age balancing:    ${BALANCE_AGE}  (bins=${N_BALANCE_AGE_BINS}, T=${BALANCE_AGE_TEMP})"
echo "  Host metadata:    ${HOST_METADATA_CSV}"
if [ "${ENCODER_TYPE}" = "pca" ]; then
  echo "  PCA cache:        ${PCA_CACHE}"
  echo "  Encoder:          pca  dim=${BOTTLENECK_DIM}"
else
  echo "  Encoder:          ${ENCODER_TYPE}  1536->[${MLP_ENCODER_HIDDEN}]->${BOTTLENECK_DIM}  (aux=${AUX_LOSS_WEIGHT}, drop=${DROPOUT}, in_drop=${INPUT_DROPOUT}, var_reg=${VARIANCE_REG_WEIGHT}, stages=${TRAINING_STAGES})"
fi
echo "  Star aggregation: ${STAR_AGGREGATION}"
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
  --n_posterior_samples ${N_POSTERIOR_SAMPLES} \
  ${ENCODER_EXTRA_ARGS}"

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
if [ "${BALANCE_AGE}" = "true" ]; then
  CMD="${CMD} --balance_age --n_balance_age_bins ${N_BALANCE_AGE_BINS} --balance_age_temp ${BALANCE_AGE_TEMP}"
fi

eval $CMD
