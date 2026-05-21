#!/bin/bash
# Latent extraction (+ UMAP) for the SPLIT-META + adversarial-head model (e27).
#
# Architecture note (why no 3-version sweep here):
#   This model was trained with --split_meta_encoders: the instrumental fields
#   (sector/camera/ccd) go to a SEPARATE head-only encoder and never enter the
#   RNN hidden states. The latent is pooled from hidden states, so it never sees
#   instrumental metadata. Therefore the "noInstr" ablation (--zero_fields
#   sector,camera,ccd) produces a latent BYTE-IDENTICAL to metaAll and is a
#   waste of compute. Only two conditions are meaningful:
#     metaAll : all metadata fed normally (the headline — probe this)
#     noMeta  : withhold ALL metadata (zeros the 9 stellar fields that DO feed
#               the latent) — optional robustness re-check only
#
# The split/adversary/instr_emb_dim/num_sectors config is read straight from the
# checkpoint (plot_umap_latent.load_model), so no extra CLI flags are needed.

# === CHANGE THIS to switch versions: metaAll | noMeta ===
META_MODE="metaAll"

MODEL_PATH="final_model/meta_split/e27/best_model.pt"
H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5 final_pretrain/timeseries_thickdisk.h5"
LATENT_BASE="final_model/meta_split/e27"

# Derive the ablation flag from META_MODE. (noInstr intentionally unsupported —
# see the architecture note above; it would equal metaAll.)
case "${META_MODE}" in
  metaAll) ZERO_FLAG="" ;;
  noMeta)  ZERO_FLAG="--zero_metadata" ;;
  *) echo "ERROR: META_MODE must be metaAll | noMeta (got '${META_MODE}')"; exit 1 ;;
esac

VARIANT_DIR="${LATENT_BASE}/${META_MODE}"
OUTPUT_DIR="${VARIANT_DIR}/umap"
mkdir -p "${VARIANT_DIR}"

# One save path per H5 file, IN THE SAME ORDER as H5_PATHS above.
SAVE_LATENTS="${VARIANT_DIR}/latents_pretrain.npz ${VARIANT_DIR}/latents_hosts.npz ${VARIANT_DIR}/latents_thickdisk.npz"

# Set LOAD_LATENTS to skip extraction and run UMAP on cached latents; leave empty
# for fresh extraction (concatenated in the order listed).
LOAD_LATENTS=""

# Strip N samples from each end of every light curve before encoding (must match
# training, which used --trim_edges 10).
TRIM_EDGES=10

# Drop N leading/trailing hidden-state timesteps before computing global_min /
# global_max only (MinGRU scan is barely averaged near each stream's start).
MINMAX_EDGE_SKIP=100

CMD="python scripts/plot_umap_latent.py \
  --age_csv_path final_pretrain/metadata.csv final_pretrain/host_all_metadata.csv \
  --output_dir ${OUTPUT_DIR} \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --use_metadata \
  ${ZERO_FLAG} \
  --batch_size 16 \
  --n_neighbors 30 \
  --min_dist 0.1 \
  --pooling_mode multiscale \
  --trim_edges ${TRIM_EDGES} \
  --minmax_edge_skip ${MINMAX_EDGE_SKIP}"

if [ -n "${LOAD_LATENTS}" ]; then
  CMD="${CMD} --load_latents ${LOAD_LATENTS}"
else
  CMD="${CMD} --model_path ${MODEL_PATH} --h5_path ${H5_PATHS}"
  if [ -n "${SAVE_LATENTS}" ]; then
    CMD="${CMD} --save_latents ${SAVE_LATENTS}"
  fi
fi

echo "META_MODE=${META_MODE} -> ${VARIANT_DIR}  (zero flag: '${ZERO_FLAG:-none}')"
eval $CMD
