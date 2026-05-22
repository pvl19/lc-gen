#!/bin/bash
# ChronoFlow leave-one-cluster-out (LOCO) — 3-way comparison, IDENTICAL fold grouping
# (shared loocv_age_folds: young bundled, big clusters individual, sparse bundled → 11 folds).
# Quantifies cluster generalization (vs the r=0.912 random-10-fold) and whether the
# latents add anything over plain rotation→age out-of-cluster.
#
#   mlp_d4 : latent_max → MLP encoder (3-stage 50/100/100) → flow   (the headline pipeline)
#   pca_d8 : latent_max → PCA(8) → flow (100 ep)                     (unsupervised, systematics-robust)
#   gyro   : log10(Prot) → flow  p(Prot | age, colour)              (gyrochronology baseline)
#
# All on the same e110 ChronoFlow latents / stars; gyro is restricted to the same star
# set via --load_latents. Read the GLOBAL r/MAE (per-fold r is undefined for single-age folds).

LAT=final_model/parallel_fixed/e110/latents_pretrain.npz
BASE=final_model/parallel_fixed/e110/loocv
COMMON="--load_latents ${LAT} --age_csv final_pretrain/all_ages.csv \
  --pooling_mode multiscale --star_aggregation latent_max --use_metadata \
  --subset_col ref --subset_val ChronoFlow --subset_csv final_pretrain/metadata.csv \
  --loga_grid_size 1000 --seed 42 --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 --n_folds 11 --loocv_age"

echo "############## MLP (3-stage 50/100/100, bottleneck 4) ##############"
python scripts/kfold_age_inference.py ${COMMON} \
  --encoder_type mlp --pca_dim 4 --mlp_encoder_hidden 128 64 \
  --training_stages three_stage --n_epochs 250 \
  --encoder_pretrain_epochs 50 --joint_finetune_epochs 100 \
  --finetune_encoder_lr_mult 0.1 --finetune_flow_lr_mult 0.1 \
  --aux_loss_weight 1.0 --dropout 0.1 --variance_reg_weight 0.25 \
  --output_dir ${BASE}/mlp_d4 || echo "  !! MLP FAILED"

echo "############## PCA (dim 8, flow-only 100 ep) ##############"
python scripts/kfold_age_inference.py ${COMMON} \
  --encoder_type pca --pca_dim 8 --training_stages joint --n_epochs 100 \
  --output_dir ${BASE}/pca_d8 || echo "  !! PCA FAILED"

echo "############## GYRO (Prot → age, same folds + star set) ##############"
python scripts/kfold_gyro_baseline.py \
  --age_csv final_pretrain/metadata.csv \
  --subset_col ref --subset_val ChronoFlow \
  --load_latents ${LAT} \
  --loocv_age --loga_grid_size 1000 --seed 42 --n_epochs 300 \
  --flow_transforms 8 --flow_hidden_dims 64 64 --n_folds 11 \
  --output_dir ${BASE}/gyro || echo "  !! GYRO FAILED"

echo ""
echo "================ LOCO COMPARISON SUMMARY ================"
python3 - <<PY
import json, os, glob
import pandas as pd, numpy as np
base = "${BASE}"
runs = [('mlp_d4','mlp_d4'), ('pca_d8','pca_d8'), ('gyro','gyro')]
print(f"{'method':10}{'N':>7}{'r (global)':>12}{'MAE_dex':>10}")
for name, d in runs:
    p = os.path.join(base, d, 'kfold_metrics.json')
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"{name:10}{m['n_samples']:>7}{m['correlation']:>12.3f}{m['mae_dex']:>10.3f}")
    else:
        print(f"{name:10}{'(missing)':>7}")
print("\nPer-fold (age-group) MAE [dex]:")
for name, d in runs:
    csv = os.path.join(base, d, 'kfold_predictions.csv')
    if not os.path.exists(csv): continue
    df = pd.read_csv(csv)
    tcol = 'log10_true_age' if 'log10_true_age' in df.columns else [c for c in df.columns if 'true' in c.lower()][0]
    pcol = 'log10_pred_median' if 'log10_pred_median' in df.columns else [c for c in df.columns if 'pred' in c.lower()][0]
    df['ae'] = (df[pcol]-df[tcol]).abs(); df['age'] = (10**df[tcol]).round(0)
    g = df.groupby('fold').agg(age_lo=('age','min'), age_hi=('age','max'), n=('ae','size'), mae=('ae','mean'))
    print(f"  --- {name} ---")
    for _,row in g.iterrows():
        rng = f"{row.age_lo:.0f}" if row.age_lo==row.age_hi else f"{row.age_lo:.0f}-{row.age_hi:.0f}"
        print(f"    {rng:>14} Myr  n={int(row.n):>4}  MAE={row.mae:.3f}")
PY
