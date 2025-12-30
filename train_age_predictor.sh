#!/bin/bash
# Train MLP to predict stellar age from latent space + BPRP0

python scripts/train_age_predictor.py \
  --model_path output/simple_rnn/models/baseline_v15_masking_real_60e.pt \
  --h5_path data/timeseries.h5 \
  --pickle_path data/star_sector_lc_formatted.pickle \
  --age_csv data/TIC_cf_data.csv \
  --output_dir output/age_predictor \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --max_length 2048 \
  --pooling_mode mean \
  --max_samples 2000 \
  --stratify_by_age \
  --n_age_bins 20 \
  --hidden_dims 128 64 32 \
  --dropout 0.2 \
  --lr 1e-3 \
  --n_epochs 100 \
  --batch_size 64 \
  --val_split 0.2
