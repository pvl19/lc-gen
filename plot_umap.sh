#!/bin/bash
# UMAP visualization of latent space colored by stellar age

python scripts/plot_umap_latent.py \
  --model_path output/performance_tests/baseline_long/model.pt \
  --h5_path data/timeseries_x.h5 \
  --age_csv_path data/phot_all.csv \
  --output_dir output/umap/v1 \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --max_length 8192 \
  --batch_size 64 \
  --n_neighbors 15 \
  --min_dist 0.1 \
  --pooling_mode multiscale \
  --max_samples 8000 \
  --stratify_by_age \
  --n_age_bins 10 \
  --bprp0_min 0 \
  --bprp0_max 4.0
