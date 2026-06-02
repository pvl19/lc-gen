#!/bin/bash
# Re-run the e50 strict LOSO and the matched gyro NLE baseline on the
# TARS-Prot subset: stars that have both an age and a TARS adopted_period.
#
# This mirrors the original loso_strict / loso_gyro_inherited runs (sendit/e50,
# all_pretrain) but restricts the stellar population to the TARS overlap and
# forces the gyro baseline to use TARS adopted_period as Prot. Both runs save
# full held-out posteriors (N=200 samples) so the log-likelihood plot can be
# rebuilt for this matched subset.
#
# Subset is encoded in final_pretrain/metadata_tars.csv (built by
# scripts/build_metadata_tars.py):
#   - `Prot` column overwritten with TARS adopted_period (NaN for non-TARS)
#   - new `tars_subset` column = 'tars' for TARS stars, '' otherwise
#
# Outputs:
#   final_model/sendit/e50/age_inference/all_pretrain/loso_strict_tars/pca4/
#   final_model/sendit/e50/age_inference/all_pretrain/loso_gyro_inherited_tars/

set -u

# --- Inputs (e50 latents) ----------------------------------------------------
LAT=final_model/sendit/e50/metaAll/latents_pretrain.npz
HOSTS=final_model/sendit/e50/metaAll/latents_hosts.npz
THICK=final_model/sendit/e50/metaAll/latents_thickdisk.npz

AGE_ROOT=final_model/sendit/e50/age_inference
PCA_CACHE=${AGE_ROOT}/shared/global_pca_d16.npz
PCA_POOL="--pca_latent_pool ${LAT} ${HOSTS} ${THICK} --pca_cache ${PCA_CACHE} --pca_cache_max_dim 16"

META=final_pretrain/metadata.csv
META_TARS=final_pretrain/metadata_tars.csv

# PCA dim matches the original loso_strict / loso_gyro_inherited runs.
PCA_DIM=4
N_POSTERIOR_SAMPLES=200

LOSO_OUT=${AGE_ROOT}/all_pretrain/loso_strict_tars/pca${PCA_DIM}
GYRO_OUT=${AGE_ROOT}/all_pretrain/loso_gyro_inherited_tars

# Refresh the TARS-subset metadata CSV (idempotent — overwrites the existing file).
echo "############## (0/2) Refresh TARS subset CSV ##############"
python scripts/build_metadata_tars.py || { echo "  !! build_metadata_tars FAILED"; exit 1; }

# ── 1) Strict LOSO on TARS subset, save posteriors ─────────────────────────
echo ""
echo "############## (1/2) e50 strict LOSO pca${PCA_DIM} on TARS subset — save posteriors ##############"
python scripts/kfold_age_inference.py \
  --load_latents ${LAT} --age_csv ${META} \
  --subset_col tars_subset --subset_val tars --subset_csv ${META_TARS} \
  --pooling_mode multiscale --star_aggregation latent_max \
  --use_metadata --loga_grid_size 1000 --seed 42 \
  --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --encoder_type pca --training_stages joint --n_epochs 100 \
  ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 10 --loso \
  --save_heldout_posteriors --n_posterior_samples ${N_POSTERIOR_SAMPLES} \
  --output_dir ${LOSO_OUT} \
  || { echo "  !! TARS strict LOSO FAILED"; exit 1; }

# ── 2) Gyro NLE, inherit LOSO folds from (1), use TARS Prot ────────────────
echo ""
echo "############## (2/2) e50 gyro LOSO-inherited (TARS Prot) — save posteriors ##############"
python scripts/kfold_gyro_loso_inherit.py \
  --age_csv ${META_TARS} \
  --metadata_csv ${META_TARS} \
  --inherit_folds_from ${LOSO_OUT}/kfold_predictions.csv \
  --flow_transforms 8 --flow_hidden_dims 64 64 \
  --lr 1e-3 --n_epochs 80 --batch_size 64 \
  --loga_grid_size 1000 --seed 42 \
  --save_heldout_posteriors --n_posterior_samples ${N_POSTERIOR_SAMPLES} \
  --output_dir ${GYRO_OUT} \
  || { echo "  !! TARS gyro inherit FAILED"; exit 1; }

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "================ TARS-subset LOSO vs gyro — SUMMARY ================"
python3 - <<PY
import json, os
rows = [
    ('strict LOSO (TARS)',  "${LOSO_OUT}"),
    ('gyro inherit (TARS)', "${GYRO_OUT}"),
]
print(f"{'run':24}{'N':>7}{'r':>9}{'MAE':>9}")
for name, d in rows:
    p = os.path.join(d, 'kfold_metrics.json')
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"{name:24}{m.get('n_samples','?'):>7}"
              f"{m.get('correlation', float('nan')):>9.3f}"
              f"{m.get('mae_dex', float('nan')):>9.3f}")
    else:
        print(f"{name:24}{'(missing)':>7}")

# Pre-TARS reference numbers (for context only)
print("\nReference (pre-TARS subset, same e50 + pca4):")
ref = [
    ('strict LOSO (all)',  '${AGE_ROOT}/all_pretrain/loso_strict/pca4'),
    ('gyro inherit (all)', '${AGE_ROOT}/all_pretrain/loso_gyro_inherited'),
]
for name, d in ref:
    p = os.path.join(d, 'kfold_metrics.json')
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"  {name:22}N={m.get('n_samples','?'):>5}  "
              f"r={m.get('correlation', float('nan')):.3f}  "
              f"MAE={m.get('mae_dex', float('nan')):.3f}")
PY

echo ""
echo "=== DONE. Posteriors saved to: ==="
echo "  ${LOSO_OUT}/heldout_posteriors.npz"
echo "  ${GYRO_OUT}/heldout_posteriors.npz"
