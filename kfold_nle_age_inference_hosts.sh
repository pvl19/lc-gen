#!/bin/bash
# K-fold normalizing-flow age inference for exoplanet hosts.
#
# Same NSF-flow architecture and 3-stage training as kfold_age_inference.sh, but
# with host-specific data loading (cached host latents + archive_ages CSV by
# GaiaDR3_ID). Outputs include the posterior median, p16, and p84 per star.
# All parameters hardcoded per project convention.

LOAD_LATENTS="final_model/parallel_fixed/e110/latents_hosts.npz"
HOST_AGE_CSV="exop_hosts/archive_ages_default.csv"
HOST_METADATA_CSV="final_pretrain/host_all_metadata.csv"

STAR_AGGREGATION="latent_max"   # latent_mean | latent_median | latent_max | latent_mean_std
USE_MG="false"                   # also include log10(MG_quick) as 4th flow context

N_FOLDS=10
SEED=42

ENCODER_TYPE="mlp"               # pca | mlp | linear
BOTTLENECK_DIM=4
MLP_ENCODER_HIDDEN="128 64"
AUX_LOSS_WEIGHT=1.0
DROPOUT=0.1
VARIANCE_REG_WEIGHT=0.25

TRAINING_STAGES="three_stage"    # joint | two_stage | three_stage
ENCODER_PRETRAIN_EPOCHS=100
JOINT_FINETUNE_EPOCHS=100
FINETUNE_ENCODER_LR_MULT=0.001
FINETUNE_FLOW_LR_MULT=0.1

LR=1e-3
LR_DECAY_RATE=0.97
WEIGHT_DECAY=1e-4
N_EPOCHS=300
BATCH_SIZE=64

FLOW_TRANSFORMS=6
FLOW_HIDDEN_DIMS="64 64"
LOGA_GRID_SIZE=1000

TRAIN_FULL="true"

OUTPUT_DIR="final_model/parallel_fixed/e110/nle_age_hosts/${STAR_AGGREGATION}"

echo "Running k-fold NLE age inference (hosts):"
echo "  Latents cache:    ${LOAD_LATENTS}"
echo "  Host age CSV:     ${HOST_AGE_CSV}"
echo "  Host metadata:    ${HOST_METADATA_CSV}"
echo "  Star aggregation: ${STAR_AGGREGATION}"
echo "  Encoder type:     ${ENCODER_TYPE}  (bottleneck=${BOTTLENECK_DIM})"
echo "  Training stages:  ${TRAINING_STAGES}"
echo "  Folds / seed:     ${N_FOLDS} / ${SEED}"
echo "  Output:           ${OUTPUT_DIR}"

CMD="python scripts/kfold_nle_age_inference_hosts.py \
  --load_latents ${LOAD_LATENTS} \
  --host_age_csv ${HOST_AGE_CSV} \
  --host_metadata_csv ${HOST_METADATA_CSV} \
  --output_dir ${OUTPUT_DIR} \
  --star_aggregation ${STAR_AGGREGATION} \
  --n_folds ${N_FOLDS} \
  --seed ${SEED} \
  --encoder_type ${ENCODER_TYPE} \
  --bottleneck_dim ${BOTTLENECK_DIM} \
  --mlp_encoder_hidden ${MLP_ENCODER_HIDDEN} \
  --aux_loss_weight ${AUX_LOSS_WEIGHT} \
  --dropout ${DROPOUT} \
  --variance_reg_weight ${VARIANCE_REG_WEIGHT} \
  --training_stages ${TRAINING_STAGES} \
  --encoder_pretrain_epochs ${ENCODER_PRETRAIN_EPOCHS} \
  --joint_finetune_epochs ${JOINT_FINETUNE_EPOCHS} \
  --finetune_encoder_lr_mult ${FINETUNE_ENCODER_LR_MULT} \
  --finetune_flow_lr_mult ${FINETUNE_FLOW_LR_MULT} \
  --lr ${LR} \
  --lr_decay_rate ${LR_DECAY_RATE} \
  --weight_decay ${WEIGHT_DECAY} \
  --n_epochs ${N_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --flow_transforms ${FLOW_TRANSFORMS} \
  --flow_hidden_dims ${FLOW_HIDDEN_DIMS} \
  --loga_grid_size ${LOGA_GRID_SIZE}"

if [ "${USE_MG}"     = "true" ]; then CMD="${CMD} --use_mg"; fi
if [ "${TRAIN_FULL}" = "true" ]; then CMD="${CMD} --train_full"; fi

eval $CMD
