#!/bin/bash
# UMAP visualization of latent space colored by stellar age

python scripts/plot_umap_latent.py \
  --model_path output/performance_tests/cf_final/model.pt \
  --h5_path final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5 \
  --age_csv_path final_pretrain/metadata.csv final_pretrain/host_all_metadata.csv \
  --output_dir output/umap/v2-full-dataset \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --batch_size 32 \
  --n_neighbors 15 \
  --min_dist 0.1 \
  --pooling_mode multiscale
