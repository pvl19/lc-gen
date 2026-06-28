#!/bin/bash
# Per-timestep saliency for a single light curve.
# All parameters hardcoded below — edit this file to change a run.
# Outputs land in output/saliency/gaia<ID>_s<sector>/.
#
# Produces (1) IG of ||z||^2, (2) IG of the PC1 direction of a latent bank
# (optional), and (3) the analytic minGRU gate-weight w_t. See
# docs/plans/2026-06-27_saliency-mapping.md for the design.

# === Model + data ==========================================================
# Match the sendit/e50 deployment by default (final_model/sendit/e50/best_model.pt
# does not exist; the actual sendit checkpoint is at meta_mask/e50). The latent
# bank for PC1 must be extracted with the SAME checkpoint.
MODEL_PATH="final_model/meta_mask/e50/best_model.pt"
H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5 final_pretrain/timeseries_thickdisk.h5"
LATENTS_NPZ="final_model/meta_mask/e50/metaAll/latents_pretrain.npz"

# === Target star ===========================================================
# Leave both empty to pick a random star with --seed.
GAIA_ID=""        # e.g. "1843146113696239616"
TIC_ID=""         # e.g. "282358593"
SEED=0

# === IG knobs ==============================================================
N_IG_STEPS=64
BASELINE_MODE="zero"          # "zero" or "mean"

# === Pool config (MUST match the latent bank used for PCA) =================
# These mirror plot_umap_metaAll.sh.
APPLY_HEAD_NORM="true"        # use --no-apply_head_norm to flip
MINMAX_EDGE_SKIP=100
GLOB_MODE="uniform"
SEG_MODE="equal_count"
DIFF_WEIGHT_MODE="dt"

# === Model build ===========================================================
HIDDEN_SIZE=64
DIRECTION="bi"
MODE="parallel"
USE_METADATA="true"
USE_CONV_CHANNELS="false"
TRIM_EDGES=10                 # must match training

OUTPUT_DIR="output/saliency"

CMD="python scripts/compute_saliency.py \
  --model_path ${MODEL_PATH} \
  --h5_paths ${H5_PATHS} \
  --seed ${SEED} \
  --n_ig_steps ${N_IG_STEPS} \
  --baseline_mode ${BASELINE_MODE} \
  --minmax_edge_skip ${MINMAX_EDGE_SKIP} \
  --glob_mode ${GLOB_MODE} \
  --seg_mode ${SEG_MODE} \
  --diff_weight_mode ${DIFF_WEIGHT_MODE} \
  --hidden_size ${HIDDEN_SIZE} \
  --direction ${DIRECTION} \
  --mode ${MODE} \
  --trim_edges ${TRIM_EDGES} \
  --output_dir ${OUTPUT_DIR}"

if [ -n "${GAIA_ID}" ]; then CMD="${CMD} --gaia_id ${GAIA_ID}"; fi
if [ -n "${TIC_ID}" ];  then CMD="${CMD} --tic_id ${TIC_ID}"; fi
if [ "${USE_METADATA}"      = "true" ]; then CMD="${CMD} --use_metadata"; fi
if [ "${USE_METADATA}"      = "false" ]; then CMD="${CMD} --no-use_metadata"; fi
if [ "${USE_CONV_CHANNELS}" = "true" ]; then CMD="${CMD} --use_conv_channels"; fi
if [ "${APPLY_HEAD_NORM}"   = "false" ]; then CMD="${CMD} --no-apply_head_norm"; fi
if [ -n "${LATENTS_NPZ}" ] && [ -f "${LATENTS_NPZ}" ]; then
  CMD="${CMD} --latents_npz ${LATENTS_NPZ}"
else
  echo "[saliency.sh] LATENTS_NPZ not provided / not found — skipping PC1 target."
fi

echo "${CMD}"
eval $CMD

# Now make the figures. compute_saliency.py writes one subdir per star; pick up
# the most recently created one.
LAST_DIR=$(ls -td ${OUTPUT_DIR}/*/ 2>/dev/null | head -1)
if [ -n "${LAST_DIR}" ]; then
  NPZ="${LAST_DIR%/}/attribution.npz"
  echo "[saliency.sh] plotting ${NPZ}"
  python scripts/plot_saliency.py --input_npz "${NPZ}" --target norm
  if [ -n "${LATENTS_NPZ}" ] && [ -f "${LATENTS_NPZ}" ]; then
    python scripts/plot_saliency.py --input_npz "${NPZ}" --target pc1
  fi
fi
