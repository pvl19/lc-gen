#!/bin/bash
# Plot light-curve reconstructions from the flow head at a given offset k.
# All parameters hardcoded below — edit this file to change a run.
# Outputs land in output/reconstructions/.

MODEL_PATH="final_model/final_parallel_e22/model.pt"
H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5"

# Selection: leave GAIA_ID and TIC_ID empty for random selection of NUM_EXAMPLES stars.
GAIA_ID=""        # e.g. "1843146113696239616"
TIC_ID=""         # e.g. "282358593"
NUM_EXAMPLES=3

OFFSET=8          # flow prediction offset k (timesteps)
SEED=0
N_FLOW_SAMPLES=64

# Must match the checkpoint.
HIDDEN_SIZE=64
DIRECTION="bi"
MODE="parallel"
USE_METADATA="true"
USE_CONV_CHANNELS="false"
CONV_ENCODER_TYPE="unet"

# Block-mask params (mirror training-time masking). Set MASK_PORTION=0 to disable.
MASK_PORTION=0.0
MASK_MIN_SIZE=2
MASK_MAX_SIZE=720

# Must match the AE training-time trim_edges value, otherwise the model sees
# out-of-distribution inputs at the sequence edges.
TRIM_EDGES=10

OUTPUT_DIR="output/reconstructions"

CMD="python scripts/plot_reconstructions.py \
  --model_path ${MODEL_PATH} \
  --h5_paths ${H5_PATHS} \
  --num_examples ${NUM_EXAMPLES} \
  --offset ${OFFSET} \
  --seed ${SEED} \
  --n_flow_samples ${N_FLOW_SAMPLES} \
  --hidden_size ${HIDDEN_SIZE} \
  --direction ${DIRECTION} \
  --mode ${MODE} \
  --mask_portion ${MASK_PORTION} \
  --mask_min_size ${MASK_MIN_SIZE} \
  --mask_max_size ${MASK_MAX_SIZE} \
  --trim_edges ${TRIM_EDGES} \
  --output_dir ${OUTPUT_DIR}"

if [ -n "${GAIA_ID}" ]; then
  CMD="${CMD} --gaia_id ${GAIA_ID}"
fi
if [ -n "${TIC_ID}" ]; then
  CMD="${CMD} --tic_id ${TIC_ID}"
fi
if [ "${USE_METADATA}" = "true" ]; then
  CMD="${CMD} --use_metadata"
fi
if [ "${USE_CONV_CHANNELS}" = "true" ]; then
  CMD="${CMD} --use_conv_channels --conv_encoder_type ${CONV_ENCODER_TYPE}"
fi

echo "${CMD}"
eval $CMD
