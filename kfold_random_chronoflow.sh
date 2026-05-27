#!/bin/bash
# Test 2 (ChronoFlow): per-light-curve random star-disjoint kfold — latent vs gyro.
# Apples-to-apples context for the LOCO (cluster-disjoint) and LOSO (sector-disjoint)
# numbers on the same N=2,470 ChronoFlow population with isochrone ages.
#
# LATENT: --star_aggregation predict_mean
#   - Trains the age flow per-light-curve (each (star, sector) row independently).
#   - star_level_split=True is forced in this mode → folds partition by Gaia ID, so
#     all sectors of a star stay in one fold (no within-star leakage).
#   - Per-sector predictions are saved; final reported metric is per-star (averaged
#     per-sector predictions). The CSV lets us compute either-granularity post-hoc.
#
# GYRO: kfold_gyro_baseline.py with --n_folds 10, NO --loocv_age.
#   - One Prot per star → per-star training + prediction, star-disjoint by
#     construction. --load_latents filters to stars that also have light curves
#     (same star population as the latent run).
#
# Reference points already in hand (same N=2,470 ChronoFlow, sendit/e50, dim 4):
#   LOCO (cluster-disjoint, latent):  r=0.709  MAE=0.457
#   LOCO (cluster-disjoint, gyro):    r=0.498  MAE=0.553   (from e110 era)
#   LOSO (sector-disjoint, latent):   r=0.732  MAE=0.473

LAT=final_model/sendit/e50/metaAll/latents_pretrain.npz
HOSTS=final_model/sendit/e50/metaAll/latents_hosts.npz
THICK=final_model/sendit/e50/metaAll/latents_thickdisk.npz
AGE_ROOT=final_model/sendit/e50/age_inference
PCA_CACHE=${AGE_ROOT}/shared/global_pca_d16.npz
SWEEP_LOCO=${AGE_ROOT}/chronoflow/loco
BASE_LAT=${AGE_ROOT}/chronoflow/random_latent
BASE_GYRO=${AGE_ROOT}/chronoflow/random_gyro

PCA_POOL="--pca_latent_pool ${LAT} ${HOSTS} ${THICK} --pca_cache ${PCA_CACHE} --pca_cache_max_dim 16"

# Pick PCA dim from sweep (best LOCO r); fallback 8.
PCA_DIM=$(python3 - <<PY
import json, os
sweep_loco = "${SWEEP_LOCO}"
best_d, best_r = None, -1e9
for d in (4, 8, 16):
    p = os.path.join(sweep_loco, f"pca{d}", "kfold_metrics.json")
    if os.path.exists(p):
        r = json.load(open(p)).get("correlation", -1e9)
        if r > best_r:
            best_r, best_d = r, d
print(best_d if best_d is not None else 8)
PY
)

# --- 1) Latent random star-disjoint kfold (predict_mean, ChronoFlow) -----
echo "############## LATENT random (predict_mean, ChronoFlow, PCA dim ${PCA_DIM}) ##############"
python scripts/kfold_age_inference.py \
  --load_latents ${LAT} --age_csv final_pretrain/metadata.csv --override_ages_from_csv \
  --subset_col ref --subset_val ChronoFlow --subset_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation predict_mean --use_metadata \
  --loga_grid_size 1000 --seed 42 --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --encoder_type pca --training_stages joint --n_epochs 100 \
  ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 10 \
  --output_dir ${BASE_LAT}/pca${PCA_DIM} || echo "  !! LATENT random FAILED"

# --- 2) Gyro random star-disjoint kfold (ChronoFlow) ---------------------
echo ""
echo "############## GYRO random (ChronoFlow) ##############"
python scripts/kfold_gyro_baseline.py \
  --age_csv final_pretrain/metadata.csv \
  --subset_col ref --subset_val ChronoFlow \
  --load_latents ${LAT} \
  --loga_grid_size 1000 --seed 42 --n_epochs 300 \
  --flow_transforms 8 --flow_hidden_dims 64 64 --n_folds 10 \
  --output_dir ${BASE_GYRO} || echo "  !! GYRO random FAILED"

# --- Summary -------------------------------------------------------------
echo ""
echo "================ CHRONOFLOW FULL COMPARISON (N=2,470, sendit/e50, dim ${PCA_DIM}) ================"
python3 - <<PY
import json, os
runs = [
    ("LOCO  (cluster-disjoint, latent)",  f"${SWEEP_LOCO}/pca${PCA_DIM}/kfold_metrics.json"),
    ("LOCO  (cluster-disjoint, gyro)",    f"${AGE_ROOT}/chronoflow/loco_gyro_reference/kfold_metrics.json"),
    ("LOSO  (sector-disjoint, latent)",   f"${AGE_ROOT}/chronoflow/loso/pca${PCA_DIM}/kfold_metrics.json"),
    ("RAND  (random per-LC, latent)",     f"${BASE_LAT}/pca${PCA_DIM}/kfold_metrics.json"),
    ("RAND  (random per-star, gyro)",     "${BASE_GYRO}/kfold_metrics.json"),
]
print(f"{'run':>36}{'N':>8}{'r':>9}{'MAE':>9}")
for name, p in runs:
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"{name:>36}{m['n_samples']:>8}{m['correlation']:>9.3f}{m['mae_dex']:>9.3f}")
    else:
        print(f"{name:>36}{'(missing)':>8}  ({p})")
PY
