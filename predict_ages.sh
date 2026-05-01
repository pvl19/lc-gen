#!/bin/bash
# Infer ages for exoplanet host stars using a trained k-fold NLE model.
#
# The encoder type (pca/mlp/linear) and training configuration (joint/two_stage/
# three_stage) are read automatically from NLE_DIR/config.json — no manual
# configuration needed. Just point NLE_DIR at any kfold output folder.
#
# Model selection: auto-detects full_model.pt (single model trained on all
# labeled stars) if present; otherwise falls back to kfold_models.pt (ensemble).
# Override with --model_file <filename> if needed.
#
# Latents workflow:
#   First run:  set LOAD_LATENTS="" and provide AE_MODEL + H5_PATH
#               → latents extracted and saved to SAVE_LATENTS
#   Later runs: set LOAD_LATENTS to the saved .npz to skip AE inference

NLE_DIR="final_model/parallel_fixed/e110/age-inference-multiscale-latent_max"
AE_MODEL="final_model/parallel_fixed/e110/best_model.pt"
H5_PATH="exop_hosts/lc_data.h5"
OUTPUT_DIR="final_model/parallel_fixed/e110/exop_host_ages"
LOAD_LATENTS="final_model/parallel_fixed/e110/latents_hosts.npz"   # set to "" to re-extract
SAVE_LATENTS=""   # e.g. exop_hosts/exop_host_ages-v3/latents.npz

# Used to fill NaN BPRP0/BPRP0_err in caches saved before the host H5 Gaia ID repair.
METADATA_CSV="final_pretrain/host_all_metadata.csv"

POOLING_MODE="multiscale"
BATCH_SIZE=32
NLE_BATCH_SIZE=256

echo "Running exoplanet host age inference:"
echo "  NLE dir:      ${NLE_DIR}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Load latents: ${LOAD_LATENTS:-'(extract from AE)'}"

CMD="python scripts/predict_ages.py \
  --nle_dir ${NLE_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --pooling_mode ${POOLING_MODE} \
  --batch_size ${BATCH_SIZE} \
  --nle_batch_size ${NLE_BATCH_SIZE}"

if [ -n "${LOAD_LATENTS}" ]; then
  CMD="${CMD} --load_latents ${LOAD_LATENTS}"
else
  CMD="${CMD} --ae_model ${AE_MODEL} --h5_path ${H5_PATH}"
fi

if [ -n "${METADATA_CSV}" ]; then
  CMD="${CMD} --metadata_csv ${METADATA_CSV}"
fi

if [ -n "${SAVE_LATENTS}" ]; then
  CMD="${CMD} --save_latents ${SAVE_LATENTS}"
fi

eval $CMD
