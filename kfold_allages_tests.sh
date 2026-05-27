#!/bin/bash
# Tests 2 + 3 on the ALL-LABELED population (pretrain + hosts).
#
# Hosts contribute only the ~192 exoplanet hosts that happen to also be in
# metadata.csv (i.e. cluster-member hosts); the rest of host_all_metadata.csv
# lacks an age column and is filtered out by the cache's NaN-age guard. So
# "all-ages incl. hosts" here means N ≈ 9,413 labeled stars (9,221 pretrain +
# 192 hosts). Hosts have no Prot column at all → excluded from the gyro run.
#
# Runs (sequential, one heavy job at a time):
#   1. LATENT RANDOM (predict_mean, star-disjoint, all labeled)
#   2. GYRO   RANDOM (pretrain only, all labeled with Prot)
#   3. LATENT LOSO   (latent_max, sector-disjoint, balanced folds, all labeled)
#
# Compares to the matched ChronoFlow numbers we already have:
#   ChronoFlow LOCO latent: r=0.709, gyro: r=0.498
#   ChronoFlow LOSO latent: r=0.732
#   ChronoFlow RAND latent: r=0.847, gyro: r=0.503
#
# The all-ages numbers will be lower because the non-ChronoFlow ages
# (gyro-derived, M-dwarf rotation surveys) have higher per-star noise.

LAT=final_model/sendit/e50/metaAll/latents_pretrain.npz
HOSTS=final_model/sendit/e50/metaAll/latents_hosts.npz
THICK=final_model/sendit/e50/metaAll/latents_thickdisk.npz
AGE_ROOT=final_model/sendit/e50/age_inference
PCA_CACHE=${AGE_ROOT}/shared/global_pca_d16.npz
SWEEP_LOCO=${AGE_ROOT}/chronoflow/loco

BASE_RAND_LAT=${AGE_ROOT}/all_pretrain_plus_hosts/random_latent
BASE_RAND_GYRO=${AGE_ROOT}/all_pretrain_plus_hosts/random_gyro
BASE_LOSO_LAT=${AGE_ROOT}/all_pretrain_plus_hosts/loso

PCA_POOL="--pca_latent_pool ${LAT} ${HOSTS} ${THICK} --pca_cache ${PCA_CACHE} --pca_cache_max_dim 16"

# Pick PCA dim from the sweep (best LOCO r); fallback 8.
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

# ── 1) Latent random (predict_mean, pretrain+hosts) ──────────────────────
echo "############## LATENT random (predict_mean, pretrain+hosts, PCA dim ${PCA_DIM}) ##############"
python scripts/kfold_age_inference.py \
  --load_latents ${LAT} ${HOSTS} \
  --age_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation predict_mean --use_metadata \
  --loga_grid_size 1000 --seed 42 --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --encoder_type pca --training_stages joint --n_epochs 100 \
  ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 10 \
  --output_dir ${BASE_RAND_LAT}/pca${PCA_DIM} || echo "  !! LATENT random FAILED"

# ── 2) Gyro random (pretrain only; hosts have no Prot) ───────────────────
echo ""
echo "############## GYRO random (pretrain only, all labeled with Prot) ##############"
python scripts/kfold_gyro_baseline.py \
  --age_csv final_pretrain/metadata.csv \
  --load_latents ${LAT} \
  --loga_grid_size 1000 --seed 42 --n_epochs 300 \
  --flow_transforms 8 --flow_hidden_dims 64 64 --n_folds 10 \
  --output_dir ${BASE_RAND_GYRO} || echo "  !! GYRO random FAILED"

# ── 3) Latent LOSO (latent_max, pretrain+hosts) ──────────────────────────
echo ""
echo "############## LATENT LOSO (latent_max, pretrain+hosts, PCA dim ${PCA_DIM}) ##############"
python scripts/kfold_age_inference.py \
  --load_latents ${LAT} ${HOSTS} \
  --age_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation latent_max --use_metadata \
  --loga_grid_size 1000 --seed 42 --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --encoder_type pca --training_stages joint --n_epochs 100 \
  ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 10 --loso \
  --output_dir ${BASE_LOSO_LAT}/pca${PCA_DIM} || echo "  !! LATENT LOSO FAILED"

# ── Summary: all-ages alongside ChronoFlow references ────────────────────
echo ""
echo "================ ALL-AGES vs CHRONOFLOW (sendit/e50, dim ${PCA_DIM}) ================"
python3 - <<PY
import json, os
runs = [
    # ChronoFlow (N≈2,470) references for context
    ("CHRONOFLOW LOCO latent", f"${SWEEP_LOCO}/pca${PCA_DIM}/kfold_metrics.json"),
    ("CHRONOFLOW LOCO gyro",   f"${AGE_ROOT}/chronoflow/loco_gyro_reference/kfold_metrics.json"),
    ("CHRONOFLOW LOSO latent", f"${AGE_ROOT}/chronoflow/loso/pca${PCA_DIM}/kfold_metrics.json"),
    ("CHRONOFLOW RAND latent", f"${AGE_ROOT}/chronoflow/random_latent/pca${PCA_DIM}/kfold_metrics.json"),
    ("CHRONOFLOW RAND gyro",   f"${AGE_ROOT}/chronoflow/random_gyro/kfold_metrics.json"),
    # All-ages (this run)
    ("ALL-AGES   RAND latent", f"${BASE_RAND_LAT}/pca${PCA_DIM}/kfold_metrics.json"),
    ("ALL-AGES   RAND gyro",   "${BASE_RAND_GYRO}/kfold_metrics.json"),
    ("ALL-AGES   LOSO latent", f"${BASE_LOSO_LAT}/pca${PCA_DIM}/kfold_metrics.json"),
]
print(f"{'run':>30}{'N':>8}{'r':>9}{'MAE':>9}")
print("-"*56)
for name, p in runs:
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"{name:>30}{m['n_samples']:>8}{m['correlation']:>9.3f}{m['mae_dex']:>9.3f}")
    else:
        print(f"{name:>30}{'(missing)':>8}  ({p})")
PY
