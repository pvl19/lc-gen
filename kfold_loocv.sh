#!/bin/bash
# Leave-one-cluster-out (LOCO) age inference on ChronoFlow — apples-to-apples
# baseline vs the r=0.912 random-10-fold headline.
#
# Mirrors EXACTLY the config that produced
#   final_model/parallel_fixed/e110/age-inference-cfonly-multiscale-latent_max/ (r=0.912)
# but swaps the split to leave-one-AGE-out (--loocv_age): each unique ChronoFlow
# isochrone age (≈ one cluster) is held out one at a time. The drop from 0.912 to
# the LOCO r/MAE = how much of the headline was cluster-memorization vs real
# cluster generalization. Same e110 latents (old pooling — the 0.912 latents), same
# latent_max aggregation, same 3-stage MLP head.
#
# Notes:
#   - --pca_h5_paths is omitted: it's a no-op in --load_latents mode (the 0.912 run
#     normalized from the ChronoFlow subset itself), so this matches it.
#   - --train_full omitted: LOCO is a diagnostic, no deployment model needed.
#   - n_folds is OVERRIDDEN by --loocv_age (one fold per unique age), so its value
#     here is irrelevant.
#   - Read the GLOBAL r/MAE in kfold_metrics.json (per-fold r is undefined — each
#     held-out fold is a single age).
#   - age_Myr cluster proxy: swap to true cluster labels later for strict LOCO.

OUTPUT_DIR="final_model/parallel_fixed/e110/age-inference-cfonly-multiscale-latent_max-LOOCV"

python scripts/kfold_age_inference.py \
  --load_latents final_model/parallel_fixed/e110/latents_pretrain.npz \
  --age_csv final_pretrain/all_ages.csv \
  --output_dir "${OUTPUT_DIR}" \
  --pooling_mode multiscale \
  --star_aggregation latent_max \
  --encoder_type mlp \
  --pca_dim 4 \
  --mlp_encoder_hidden 128 64 \
  --n_folds 10 \
  --lr 1e-3 \
  --lr_decay_rate 0.97 \
  --n_epochs 300 \
  --batch_size 64 \
  --flow_transforms 6 \
  --flow_hidden_dims 64 64 \
  --loga_grid_size 1000 \
  --seed 42 \
  --use_metadata \
  --aux_loss_weight 1.0 \
  --dropout 0.1 \
  --variance_reg_weight 0.25 \
  --training_stages three_stage \
  --encoder_pretrain_epochs 100 \
  --joint_finetune_epochs 100 \
  --finetune_encoder_lr_mult 0.1 \
  --finetune_flow_lr_mult 0.1 \
  --subset_col ref --subset_val ChronoFlow --subset_csv final_pretrain/metadata.csv \
  --loocv_age
