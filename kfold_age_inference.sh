#!/bin/bash
# K-fold cross-validation for age prediction from latent space + BPRP0
# Each star gets a prediction from a model that never saw it during training

python scripts/kfold_age_inference.py \
  --model_path output/simple_rnn/models/baseline_v15_masking_real_60e.pt \
  --h5_path data/timeseries.h5 \
  --pickle_path data/star_sector_lc_formatted.pickle \
  --age_csv data/TIC_cf_data.csv \
  --output_dir output/age_predictor/kfold \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --max_length 2048 \
  --pooling_mode mean \
  --stratify_by_age \
  --n_age_bins 20 \
  --n_folds 10 \
  --hidden_dims 128 64 32 \
  --dropout 0.2 \
  --lr 1e-3 \
  --n_epochs 100 \
  --batch_size 64 \
  --seed 42
