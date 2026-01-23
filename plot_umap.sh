#!/bin/bash
# UMAP visualization of latent space colored by stellar age

python scripts/plot_umap_latent.py \
  --model_path output/simple_rnn/models/baseline_v17_real_final.pt \
  --h5_path data/timeseries.h5 \
  --pickle_path data/star_sector_lc_formatted.pickle \
  --age_csv_path data/TIC_cf_data.csv \
  --output_dir output/simple_rnn/umap_plots \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --use_metadata \
  --max_length 2048 \
  --batch_size 32 \
  --n_neighbors 15 \
  --min_dist 0.1 \
  --pooling_mode mean \
  --max_samples 7911 \
  --stratify_by_age \
  --n_age_bins 10 \
  --bprp0_min 0 \
  --bprp0_max 4.0
