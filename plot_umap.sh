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
SAVE_LATENTS="final_model/parallel_fixed/e110/latents_pretrain.npz final_model/parallel_fixed/e110/latents_hosts.npz"
MODEL_PATH="final_model/parallel_fixed/e110/best_model.pt"
H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5"

OUTPUT_DIR="final_model/parallel_fixed/e110/umap"

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
  --pooling_mode multiscale"

if [ -n "${LOAD_LATENTS}" ]; then
  CMD="${CMD} --load_latents ${LOAD_LATENTS}"
else
  CMD="${CMD} --model_path ${MODEL_PATH} --h5_path ${H5_PATHS}"
  if [ -n "${SAVE_LATENTS}" ]; then
    CMD="${CMD} --save_latents ${SAVE_LATENTS}"
  fi
fi

eval $CMD
