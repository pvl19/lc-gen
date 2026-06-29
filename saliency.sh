#!/bin/bash
# Per-timestep saliency for a single light curve.
# All parameters hardcoded below — edit this file to change a run.
# Outputs land in output/saliency/gaia<ID>_s<sector>/.
#
# Produces, per star:
#   (1) Integrated Gradients of ||z||^2 and (z-mean)·PC1 w.r.t. (flux, flux_err).
#       N=N_IG_STEPS fade-in points from zero baseline to the real input.
#   (2) Single-point whole-timestep occlusion: f(z_full) - f(z_occluded_t)
#       under mask[t]=0 (the encoder's trained missing-data behavior), for
#       both targets. Per-channel ablations are opt-in via
#       RUN_OCCLUSION_PER_CHANNEL.
#   (3) The analytic minGRU write-gate magnitude w_t = mean_d sigmoid(W_z x_t)_d.
#   (4) 3-panel figures for each (method × target) combination, into the same
#       per-star subdirectory.
# See docs/plans/2026-06-27_saliency-mapping.md for the design.

# === Model + data ==========================================================
# All latent banks must be extracted with the SAME checkpoint as MODEL_PATH.
# Concatenating pretrain + hosts + thickdisk gives a PC1 representative of the
# full deployment population rather than the labeled (cluster-heavy) subset.
MODEL_PATH="final_model/sendit/e100/best_model.pt"
H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5 final_pretrain/timeseries_thickdisk.h5"
LATENTS_NPZ="final_model/sendit/e100/latents_pretrain.npz final_model/sendit/e100/latents_hosts.npz final_model/sendit/e100/latents_thickdisk.npz"

# === Target star ===========================================================
# Leave both empty to pick a random star with --seed.
GAIA_ID=""        # e.g. "1843146113696239616"
TIC_ID=""         # e.g. "282358593"
SEED=0

# === IG knobs ==============================================================
N_IG_STEPS=32
BASELINE_MODE="zero"          # "zero" or "mean"

# === Occlusion knobs =======================================================
# Default: whole-timestep ablation only (mask[t]=0 → encoder's trained
# missing-data behavior). Single attribution per timestep, cleanest reading
# of "what does the model lose without this point". At BATCH=64 a typical
# L≈11K star takes ~4-6 min on CPU.
# Setting RUN_OCCLUSION_PER_CHANNEL=true adds the flux-only and err-only
# modes (3× cost). The per-channel ablations feed synthetic inputs (e.g.
# flux=0, flux_err=normal) the encoder never saw at training — useful for
# specific science questions, but interpret with caution.
RUN_OCCLUSION="true"
RUN_OCCLUSION_PER_CHANNEL="false"
OCCLUSION_BATCH_SIZE=64

# === Mirror-symmetry diagnostic (decompose forward / backward) =============
# When true, runs two extra IGs against ||z_fwd||^2 and ||z_bwd||^2 (each is
# the pool of only one direction's hidden states). Reports the Pearson corr
# between s_fwd[t] and time-reversed s_bwd[t]: corr ≈ +1 means the trained
# forward/backward encoders are mirror-symmetric (so bidirectionality should
# cancel time asymmetry); low or negative corr means they've diverged.
# Adds 2*N_IG_STEPS forward+backward passes.
DECOMPOSE_DIRECTIONS="false"

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
  --occlusion_batch_size ${OCCLUSION_BATCH_SIZE} \
  --output_dir ${OUTPUT_DIR}"

if [ -n "${GAIA_ID}" ]; then CMD="${CMD} --gaia_id ${GAIA_ID}"; fi
if [ -n "${TIC_ID}" ];  then CMD="${CMD} --tic_id ${TIC_ID}"; fi
if [ "${USE_METADATA}"      = "true" ]; then CMD="${CMD} --use_metadata"; fi
if [ "${USE_METADATA}"      = "false" ]; then CMD="${CMD} --no-use_metadata"; fi
if [ "${USE_CONV_CHANNELS}" = "true" ]; then CMD="${CMD} --use_conv_channels"; fi
if [ "${APPLY_HEAD_NORM}"   = "false" ]; then CMD="${CMD} --no-apply_head_norm"; fi
if [ "${RUN_OCCLUSION}"             = "false" ]; then CMD="${CMD} --no-run_occlusion"; fi
if [ "${RUN_OCCLUSION_PER_CHANNEL}"  = "true"  ]; then CMD="${CMD} --run_occlusion_per_channel"; fi
if [ "${DECOMPOSE_DIRECTIONS}"       = "true"  ]; then CMD="${CMD} --decompose_directions"; fi
# LATENTS_NPZ may be one or several space-separated paths; word-splits into
# multiple --latents_npz args. The python script concatenates the banks before
# fitting PC1.
LATENTS_OK="true"
for p in ${LATENTS_NPZ}; do
  if [ ! -f "${p}" ]; then
    echo "[saliency.sh] latent bank missing: ${p} — skipping PC1 target."
    LATENTS_OK="false"
    break
  fi
done
if [ -n "${LATENTS_NPZ}" ] && [ "${LATENTS_OK}" = "true" ]; then
  CMD="${CMD} --latents_npz ${LATENTS_NPZ}"
fi

echo "${CMD}"
eval $CMD

# Now make the figures. compute_saliency.py writes one subdir per star; pick up
# the most recently created one.
LAST_DIR=$(ls -td ${OUTPUT_DIR}/*/ 2>/dev/null | head -1)
if [ -n "${LAST_DIR}" ]; then
  NPZ="${LAST_DIR%/}/attribution.npz"
  echo "[saliency.sh] plotting ${NPZ}"
  python scripts/plot_saliency.py --input_npz "${NPZ}" --method ig --target norm
  if [ -n "${LATENTS_NPZ}" ] && [ "${LATENTS_OK}" = "true" ]; then
    python scripts/plot_saliency.py --input_npz "${NPZ}" --method ig --target pc1
  fi
  if [ "${RUN_OCCLUSION}" = "true" ]; then
    python scripts/plot_saliency.py --input_npz "${NPZ}" --method occlusion --target norm
    if [ -n "${LATENTS_NPZ}" ] && [ "${LATENTS_OK}" = "true" ]; then
      python scripts/plot_saliency.py --input_npz "${NPZ}" --method occlusion --target pc1
    fi
  fi
fi
