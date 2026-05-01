#!/bin/bash
# K-fold MLP age regression for exoplanet hosts.
#
# Drops the normalizing-flow posterior used in kfold_age_inference.sh and uses a
# plain MLP regressor over the encoder latents (+ optional metadata) → log10(age_Myr).
# All parameters hardcoded per project convention.

LOAD_LATENTS="final_model/parallel_fixed/e110/latents_hosts.npz"
HOST_AGE_CSV="exop_hosts/archive_ages_default.csv"
HOST_METADATA_CSV="final_pretrain/host_all_metadata.csv"

STAR_AGGREGATION="latent_max"   # latent_mean | latent_median | latent_max | latent_mean_std
USE_METADATA="true"              # concat BPRP0 + log10(BPRP0_err) to latents
USE_MG="false"                   # also include log10(MG_quick)

N_FOLDS=10
N_AGE_BINS=10
SEED=42

MLP_HIDDEN="256 128 64"
DROPOUT=0.1
LR=1e-3
WEIGHT_DECAY=1e-4
N_EPOCHS=300
BATCH_SIZE=64
PATIENCE=40
LOSS_FN="mse"                    # mse | huber

TRAIN_FULL="true"
VERBOSE="false"

OUTPUT_DIR="final_model/parallel_fixed/e110/mlp_age_hosts/${STAR_AGGREGATION}"

echo "Running k-fold MLP age inference (hosts):"
echo "  Latents cache:    ${LOAD_LATENTS}"
echo "  Host age CSV:     ${HOST_AGE_CSV}"
echo "  Host metadata:    ${HOST_METADATA_CSV}"
echo "  Star aggregation: ${STAR_AGGREGATION}"
echo "  Use metadata:     ${USE_METADATA}  (use_mg=${USE_MG})"
echo "  MLP hidden:       ${MLP_HIDDEN}"
echo "  Folds / seed:     ${N_FOLDS} / ${SEED}"
echo "  Epochs / patience:${N_EPOCHS} / ${PATIENCE}"
echo "  Output:           ${OUTPUT_DIR}"

CMD="python scripts/kfold_mlp_age_inference.py \
  --load_latents ${LOAD_LATENTS} \
  --host_age_csv ${HOST_AGE_CSV} \
  --host_metadata_csv ${HOST_METADATA_CSV} \
  --output_dir ${OUTPUT_DIR} \
  --star_aggregation ${STAR_AGGREGATION} \
  --n_folds ${N_FOLDS} \
  --n_age_bins ${N_AGE_BINS} \
  --seed ${SEED} \
  --mlp_hidden ${MLP_HIDDEN} \
  --dropout ${DROPOUT} \
  --lr ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --n_epochs ${N_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --patience ${PATIENCE} \
  --loss_fn ${LOSS_FN}"

if [ "${USE_METADATA}" = "true" ]; then CMD="${CMD} --use_metadata"; fi
if [ "${USE_MG}"       = "true" ]; then CMD="${CMD} --use_mg"; fi
if [ "${TRAIN_FULL}"   = "true" ]; then CMD="${CMD} --train_full"; fi
if [ "${VERBOSE}"      = "true" ]; then CMD="${CMD} --verbose"; fi

eval $CMD
