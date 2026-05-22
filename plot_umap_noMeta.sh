#!/bin/bash
# UMAP visualization + latent extraction for the metadata-masking (e50) model.
#
# Sector-leakage diagnostic: extract latents under three metadata conditions and
# probe each. Set META_MODE below, run the script, then change META_MODE and
# re-run — once per version. Each run writes one latent file per H5 category
# (pretrain, hosts, thickdisk) into final_model/meta_mask/e50/<META_MODE>/.
#
#   metaAll : all metadata fed normally
#   noInstr : withhold sector,camera,ccd (value 0 + mask 0); keep astrophysical metadata
#   noMeta  : withhold ALL metadata (value 0 + mask 0)
#
# noInstr/noMeta set the metadata mask channel to 0 for withheld fields, matching
# how the e50 model saw missing metadata during DOROTHY-style training (so the
# ablation is in-distribution, not "a genuine zero value").

# === CHANGE THIS to switch versions: metaAll | noInstr | noMeta ===
META_MODE="noMeta"

MODEL_PATH="final_model/meta_mask/e50/best_model.pt"
H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5 final_pretrain/timeseries_thickdisk.h5"
LATENT_BASE="final_model/meta_mask/e50"

# Derive the ablation flag from META_MODE.
case "${META_MODE}" in
  metaAll) ZERO_FLAG="" ;;
  noInstr) ZERO_FLAG="--zero_fields sector,camera,ccd" ;;
  noMeta)  ZERO_FLAG="--zero_metadata" ;;
  *) echo "ERROR: META_MODE must be metaAll | noInstr | noMeta (got '${META_MODE}')"; exit 1 ;;
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
# training). 10 removes leading TESS pipeline artifacts that otherwise contaminate
# the multiscale latent via GRU hidden states.
TRIM_EDGES=10

# Drop N leading/trailing hidden-state timesteps before computing global_min /
# global_max only. The MinGRU cumulative scan is barely averaged near each
# stream's start, so a single un-smoothed sample can dominate per-feature
# min/max even after trim_edges. Mean/std/segment/diff stats are unaffected.
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
