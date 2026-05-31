#!/bin/bash
# All five latent-based age-inference tests on the FINAL sendit/e100 model,
# using the merged latents bank (sectors 97/98 split into halves, all other
# sectors unchanged). PCA dim 4 throughout — no dim sweep (e50 LOCO sweep
# picked pca4 as the winner; reusing that choice here).
#
# Tests included, in run order:
#   1. ChronoFlow LOCO   (cluster-disjoint, latent_max)        N≈2,470
#   2. ChronoFlow LOSO   (sector-disjoint, balanced LPT)        N≈2,470
#   3. ChronoFlow RAND   (per-LC star-disjoint, predict_mean)   N≈2,470
#   4. ALL-PRETRAIN LOSO relaxed (star_disjoint val, full train) N≈9,221
#   5. ALL-PRETRAIN RAND (per-LC star-disjoint, predict_mean)   N≈9,221
#
# Rotation baselines (gyro) are not rerun — they only use Prot + colour and
# are encoder-independent; the e50 gyro numbers stand.
#
# All runs share one global PCA basis fit on the e100 merged latent pool
# (pretrain + hosts + thickdisk merged with the split-half s97/s98 rows).
# The first run below creates the cache; the rest load it.

set -u  # fail loud on unset vars; do NOT use -e (we want partial-failure tolerance)

# --- Inputs (e100 merged latents) ----------------------------------------
LAT=final_model/sendit/e100/latents_pretrain_merged.npz
HOSTS=final_model/sendit/e100/latents_hosts_merged.npz
THICK=final_model/sendit/e100/latents_thickdisk_merged.npz

# --- Outputs --------------------------------------------------------------
AGE_ROOT=final_model/sendit/e100/age_inference
BASE_CF_LOCO=${AGE_ROOT}/chronoflow/loco
BASE_CF_LOSO=${AGE_ROOT}/chronoflow/loso
BASE_CF_RAND=${AGE_ROOT}/chronoflow/random_latent
BASE_AP_LOSO_REL=${AGE_ROOT}/all_pretrain/loso_relaxed
BASE_AP_RAND=${AGE_ROOT}/all_pretrain/random_latent

# Global PCA cache — fit once on the e100 merged pool at dim 16 (lower dims
# truncate from the same basis). First test below creates it; the rest load.
PCA_CACHE=${AGE_ROOT}/shared/global_pca_d16.npz
PCA_POOL="--pca_latent_pool ${LAT} ${HOSTS} ${THICK} --pca_cache ${PCA_CACHE} --pca_cache_max_dim 16"

PCA_DIM=4
mkdir -p ${AGE_ROOT}/shared

# Common training hyperparams (mirror the e50 wrappers).
COMMON_TRAIN="--use_metadata --loga_grid_size 1000 --seed 42 \
  --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --encoder_type pca --training_stages joint --n_epochs 100"

# ── 1) ChronoFlow LOCO ────────────────────────────────────────────────────
echo "############## (1/5) CHRONOFLOW LOCO — pca${PCA_DIM} ##############"
python scripts/kfold_age_inference.py \
  --load_latents ${LAT} --age_csv final_pretrain/metadata.csv --override_ages_from_csv \
  --subset_col ref --subset_val ChronoFlow --subset_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation latent_max \
  ${COMMON_TRAIN} ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 11 --loocv_age \
  --output_dir ${BASE_CF_LOCO}/pca${PCA_DIM} \
  || echo "  !! ChronoFlow LOCO FAILED"

# ── 2) ChronoFlow LOSO ────────────────────────────────────────────────────
echo ""
echo "############## (2/5) CHRONOFLOW LOSO — pca${PCA_DIM} ##############"
python scripts/kfold_age_inference.py \
  --load_latents ${LAT} --age_csv final_pretrain/metadata.csv --override_ages_from_csv \
  --subset_col ref --subset_val ChronoFlow --subset_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation latent_max \
  ${COMMON_TRAIN} ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 10 --loso \
  --output_dir ${BASE_CF_LOSO}/pca${PCA_DIM} \
  || echo "  !! ChronoFlow LOSO FAILED"

# ── 3) ChronoFlow per-star random ─────────────────────────────────────────
echo ""
echo "############## (3/5) CHRONOFLOW RAND (predict_mean) — pca${PCA_DIM} ##############"
python scripts/kfold_age_inference.py \
  --load_latents ${LAT} --age_csv final_pretrain/metadata.csv --override_ages_from_csv \
  --subset_col ref --subset_val ChronoFlow --subset_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation predict_mean \
  ${COMMON_TRAIN} ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 10 \
  --output_dir ${BASE_CF_RAND}/pca${PCA_DIM} \
  || echo "  !! ChronoFlow RAND FAILED"

# ── 4) All-pretrain LOSO relaxed ──────────────────────────────────────────
# All labeled pretrain stars (mixed catalogs). Relaxed = training = all stars
# NOT in this fold's validation set (stars touching held-out sectors are kept
# in training; only their held-out-sector rows are dropped). Defensible at
# pca4 because the bottleneck precludes per-star memorization.
echo ""
echo "############## (4/5) ALL-PRETRAIN LOSO relaxed — pca${PCA_DIM} ##############"
python scripts/kfold_age_inference.py \
  --load_latents ${LAT} --age_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation latent_max \
  ${COMMON_TRAIN} ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 10 --loso --loso_relaxed_train \
  --output_dir ${BASE_AP_LOSO_REL}/pca${PCA_DIM} \
  || echo "  !! all-pretrain LOSO relaxed FAILED"

# ── 5) All-pretrain per-star random ───────────────────────────────────────
# No --subset_val: full N≈9,221 labeled-pretrain set. No --override_ages_from_csv
# (mirrors the LOSO runs — uses the same blended cache ages so this is a
# matched-population counterpart to test (4)).
echo ""
echo "############## (5/5) ALL-PRETRAIN RAND (predict_mean) — pca${PCA_DIM} ##############"
python scripts/kfold_age_inference.py \
  --load_latents ${LAT} --age_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation predict_mean \
  ${COMMON_TRAIN} ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 10 \
  --output_dir ${BASE_AP_RAND}/pca${PCA_DIM} \
  || echo "  !! all-pretrain RAND FAILED"

# ── Final cross-test summary ──────────────────────────────────────────────
echo ""
echo "================ e100 AGE-INFERENCE SUITE — PCA dim ${PCA_DIM} ================"
python3 - <<PY
import json, os
runs = [
    ("ChronoFlow LOCO        (latent)",     "${BASE_CF_LOCO}/pca${PCA_DIM}/kfold_metrics.json"),
    ("ChronoFlow LOSO        (latent)",     "${BASE_CF_LOSO}/pca${PCA_DIM}/kfold_metrics.json"),
    ("ChronoFlow RAND        (latent)",     "${BASE_CF_RAND}/pca${PCA_DIM}/kfold_metrics.json"),
    ("ALL-PRETRAIN LOSO rel  (latent)",     "${BASE_AP_LOSO_REL}/pca${PCA_DIM}/kfold_metrics.json"),
    ("ALL-PRETRAIN RAND      (latent)",     "${BASE_AP_RAND}/pca${PCA_DIM}/kfold_metrics.json"),
    # e50 gyro baselines (encoder-independent, kept as anchors)
    ("ChronoFlow RAND        (gyro,  e50)", "final_model/sendit/e50/age_inference/chronoflow/random_gyro/kfold_metrics.json"),
    ("ChronoFlow LOCO        (gyro,  e110)", "final_model/sendit/e50/age_inference/chronoflow/loco_gyro_reference/kfold_metrics.json"),
    ("ALL-PRETRAIN RAND      (gyro,  e50)", "final_model/sendit/e50/age_inference/all_pretrain_plus_hosts/random_gyro/kfold_metrics.json"),
]
print(f"{'run':>40}{'N':>8}{'r':>9}{'MAE':>9}")
for name, p in runs:
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"{name:>40}{m['n_samples']:>8}{m['correlation']:>9.3f}{m['mae_dex']:>9.3f}")
    else:
        print(f"{name:>40}{'(missing)':>8}")
PY
