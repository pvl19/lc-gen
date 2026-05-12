#!/bin/bash
# UMAP visualization of latent space colored by stellar age.
#
# When LOAD_LATENTS is set, model loading and H5 extraction are skipped — UMAP
# runs directly on cached latents (compatible with kfold_age_inference.py caches).
# Caches are concatenated in the order listed.

# Set LOAD_LATENTS to skip extraction; leave empty for fresh extraction.
LOAD_LATENTS=""

# SAVE_LATENTS works in both modes: one path per H5 file (fresh extraction) or
# one per LOAD_LATENTS entry (re-saves each cache in canonical format with ages
# filled from CSV — useful for normalizing predict_ages.py format caches).
SAVE_LATENTS="final_model/parallel_fixed/e110_trim10/latents_pretrain.npz final_model/parallel_fixed/e110_trim10/latents_hosts.npz"
MODEL_PATH="final_model/parallel_fixed/e110/best_model.pt"
H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5"

# Strip N samples from each end of every light curve before encoding. Set to 0
# for the original behavior. 10 removes leading TESS pipeline artifacts that
# otherwise contaminate the multiscale latent via GRU hidden states.
TRIM_EDGES=10

# Drop N leading/trailing hidden-state timesteps before computing global_min /
# global_max only. The MinGRU cumulative scan is barely averaged near each
# stream's start, so a single un-smoothed sample can dominate per-feature
# min/max even after trim_edges. Mean/std/segment/diff stats are unaffected.
MINMAX_EDGE_SKIP=100

OUTPUT_DIR="final_model/parallel_fixed/e110_trim10/umap"

CMD="python scripts/plot_umap_latent.py \
  --age_csv_path final_pretrain/metadata.csv final_pretrain/host_all_metadata.csv \
  --output_dir ${OUTPUT_DIR} \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --use_metadata \
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

eval $CMD
