#!/bin/bash
# Sector-confound-robust age inference (per-sector MLP+NF).
#
# Wraps scripts/kfold_age_inference.py with the three sector mitigations. All
# parameters are hardcoded (project convention — never pass args to .sh files).
# Toggle the three MITIGATIONS below and re-run; OUTPUT_DIR encodes the active
# set so runs do not clobber each other.
#
# IMPORTANT: these mitigations need a per-sample sector, so STAR_AGGREGATION must
# be 'none' (per-sector rows) — NOT latent_max. That means results are compared
# against a per-sector star-split baseline, not the 0.912 latent_max headline.
# Run BASELINE first (all toggles off) to get that reference, then flip toggles.
#
#   1. SECTOR_SPLIT       — sector-disjoint CV: hold out whole sectors per fold so
#                           each fold predicts stars from unseen sectors (field-star
#                           generalization proxy). Drops val rows whose star also
#                           appears in a training sector (star-disjoint holdout).
#   2. BALANCE_SECTOR_AGE — flatten the (sector x age-bin) joint via weighted
#                           sampling so cluster cells stop dominating.
#   3. ADV_SECTOR_WEIGHT  — GRL sector adversary peak λ on the MLP bottleneck
#                           (0 = off). Ramped 0->λ over training. TUNE THIS: sweep
#                           e.g. 0.1 / 0.3 / 0.5 / 1.0 and watch the partial-r probe
#                           (sector probe DOWN while within-sector age signal stays
#                           UP). λ too high erodes real age signal.
#
# Workflow: run BASELINE, then SECTOR_SPLIT alone (diagnose), then layer
# BALANCE_SECTOR_AGE / ADV_SECTOR_WEIGHT and re-probe.

# ===================== MITIGATION TOGGLES (edit these) =====================
SECTOR_SPLIT="false"          # true  -> --sector_level_split (option 1)
BALANCE_SECTOR_AGE="false"    # true  -> --balance_sector_age (option 2)
ADV_SECTOR_WEIGHT="0.0"       # >0    -> --adv_sector_weight λ (option 3)
N_BALANCE_AGE_BINS=10
ADV_HIDDEN=64
# ===========================================================================

# --- Data / model (load cached per-sector latents; no model fwd needed) ---
LOAD_LATENTS="final_model/parallel_fixed/e110/latents_pretrain.npz"
AGE_CSV="final_pretrain/all_ages.csv"
PCA_H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5"

# Per-sector regime + MLP encoder (adversary attaches to the MLP bottleneck).
POOLING_MODE="multiscale"
STAR_AGGREGATION="none"
ENCODER_TYPE="mlp"

# Subset to ChronoFlow to match the headline comparison; clear SUBSET_COL for all stars.
SUBSET_COL="ref"
SUBSET_VAL="ChronoFlow"
SUBSET_CSV="final_pretrain/metadata.csv"

# --- Age-flow / encoder training config (matches the best 3-stage MLP head) ---
MLP_ENCODER_HIDDEN="128 64"
PCA_DIM=4                      # bottleneck width
AUX_LOSS_WEIGHT=1.0
DROPOUT=0.1
VARIANCE_REG_WEIGHT=0.25
TRAINING_STAGES="three_stage"
ENCODER_PRETRAIN_EPOCHS=100
JOINT_FINETUNE_EPOCHS=100
FINETUNE_ENCODER_LR_MULT=0.1
FINETUNE_FLOW_LR_MULT=0.1
N_EPOCHS=300
N_FOLDS=10

# --- Output dir tag reflects which mitigations are active ---
TAG="baseline"
[ "${SECTOR_SPLIT}" = "true" ]       && TAG="${TAG}+secsplit"
[ "${BALANCE_SECTOR_AGE}" = "true" ] && TAG="${TAG}+balance"
# bash float compare: treat any non-"0.0"/"0" as on
case "${ADV_SECTOR_WEIGHT}" in 0|0.0|0.00) ;; *) TAG="${TAG}+adv${ADV_SECTOR_WEIGHT}" ;; esac
OUTPUT_DIR="final_model/parallel_fixed/e110/sector-robust/${STAR_AGGREGATION}__${TAG}"

echo "Sector-robust age inference:"
echo "  Latents:          ${LOAD_LATENTS}"
echo "  Aggregation:      ${STAR_AGGREGATION}   Encoder: ${ENCODER_TYPE}"
echo "  sector_split:     ${SECTOR_SPLIT}"
echo "  balance_sec_age:  ${BALANCE_SECTOR_AGE} (bins=${N_BALANCE_AGE_BINS})"
echo "  adv_sector_weight:${ADV_SECTOR_WEIGHT} (hidden=${ADV_HIDDEN})"
echo "  Output:           ${OUTPUT_DIR}"

CMD="python scripts/kfold_age_inference.py \
  --load_latents ${LOAD_LATENTS} \
  --age_csv ${AGE_CSV} \
  --pca_h5_paths ${PCA_H5_PATHS} \
  --output_dir ${OUTPUT_DIR} \
  --pooling_mode ${POOLING_MODE} \
  --star_aggregation ${STAR_AGGREGATION} \
  --encoder_type ${ENCODER_TYPE} \
  --pca_dim ${PCA_DIM} \
  --mlp_encoder_hidden ${MLP_ENCODER_HIDDEN} \
  --n_folds ${N_FOLDS} \
  --lr 1e-3 \
  --lr_decay_rate 0.97 \
  --n_epochs ${N_EPOCHS} \
  --batch_size 64 \
  --flow_transforms 6 \
  --flow_hidden_dims 64 64 \
  --loga_grid_size 1000 \
  --seed 42 \
  --use_metadata \
  --aux_loss_weight ${AUX_LOSS_WEIGHT} \
  --dropout ${DROPOUT} \
  --variance_reg_weight ${VARIANCE_REG_WEIGHT} \
  --training_stages ${TRAINING_STAGES} \
  --encoder_pretrain_epochs ${ENCODER_PRETRAIN_EPOCHS} \
  --joint_finetune_epochs ${JOINT_FINETUNE_EPOCHS} \
  --finetune_encoder_lr_mult ${FINETUNE_ENCODER_LR_MULT} \
  --finetune_flow_lr_mult ${FINETUNE_FLOW_LR_MULT} \
  --n_balance_age_bins ${N_BALANCE_AGE_BINS} \
  --adv_sector_weight ${ADV_SECTOR_WEIGHT} \
  --adv_hidden ${ADV_HIDDEN}"

[ "${SECTOR_SPLIT}" = "true" ]       && CMD="${CMD} --sector_level_split"
[ "${BALANCE_SECTOR_AGE}" = "true" ] && CMD="${CMD} --balance_sector_age"

if [ -n "${SUBSET_COL}" ] && [ -n "${SUBSET_VAL}" ]; then
  CMD="${CMD} --subset_col ${SUBSET_COL} --subset_val ${SUBSET_VAL}"
  [ -n "${SUBSET_CSV}" ] && CMD="${CMD} --subset_csv ${SUBSET_CSV}"
fi

eval $CMD
