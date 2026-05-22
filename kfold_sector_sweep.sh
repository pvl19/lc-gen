#!/bin/bash
# Sector-confound mitigation SWEEP — runs the experiment matrix SEQUENTIALLY
# (one k-fold at a time, so it's memory-safe on the laptop). All params hardcoded.
#
# Strategy under test: e50 pretraining (masked metadata), per-sector age inference,
# confound addressed at the age-inference stage. Latents are the EXISTING e50 cache
# (OLD pooling) — fine for finding the generalization gap and the adversary λ knee,
# which are second-order to the pooling recipe. Re-extract e50 with the new default
# pooling (uniform/equal_count/dt) for the FINAL tuned run.
#
# Reduced N_EPOCHS (150 vs the 300 headline) for sweep turnaround — relative
# comparisons hold; bump back to 300 for the final config.
#
# Matrix: a baseline (predict_mean star-disjoint reference) + the mitigation set
# {plain secdisjoint, balance, adv 0.3/0.5/1.0} run under BOTH holdout variants
# (__sectoronly and __stardisjoint, see below). Gap vs baseline = sector reliance;
# the adversary λ sweep asks whether it closes the gap without hurting accuracy.

OUTBASE="final_model/meta_mask/e50/sector-robust/sweep"

# ALL labeled stars (no ChronoFlow subset): more single-sector field-like stars
# → robust star-disjoint sector-CV folds (cfonly is heavily multi-sector, which
# empties folds), and better matches the "applicability to other stars" goal.
# Reduced n_folds=5 / n_epochs=120 keep the all-stars sweep feasible on the laptop;
# relative comparisons hold, bump to 10/300 for the final config.
COMMON="--load_latents final_model/meta_mask/e50/metaAll/latents_pretrain.npz \
  --age_csv final_pretrain/all_ages.csv \
  --pca_h5_paths final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5 \
  --encoder_type mlp --pca_dim 4 --mlp_encoder_hidden 128 64 \
  --n_folds 5 --lr 1e-3 --lr_decay_rate 0.97 --n_epochs 120 --batch_size 64 \
  --flow_transforms 6 --flow_hidden_dims 64 64 --loga_grid_size 1000 --seed 42 \
  --use_metadata --aux_loss_weight 1.0 --dropout 0.1 --variance_reg_weight 0.25 \
  --training_stages three_stage --encoder_pretrain_epochs 40 --joint_finetune_epochs 40 \
  --finetune_encoder_lr_mult 0.1 --finetune_flow_lr_mult 0.1"

# Two holdout variants — BOTH hold the SECTOR out of training (--sector_level_split):
#   __sectoronly  : sector held out; the star MAY appear via another sector
#                   (--sector_split_keep_star_overlap). Larger holdout; the primary
#                   "sector not seen in training" reading.
#   __stardisjoint: sector AND star both held out (default star-overlap drop).
#                   Stricter (unseen star in an unseen sector); smaller holdout.
SD="--star_aggregation none --sector_level_split"
KEEP="--sector_split_keep_star_overlap"

# name : extra flags (sequential)
points=(
  "baseline:--star_aggregation predict_mean"
  # --- sector-only (primary) ---
  "secdisjoint__sectoronly:${SD} ${KEEP}"
  "balance__sectoronly:${SD} ${KEEP} --balance_sector_age"
  "adv0.3__sectoronly:${SD} ${KEEP} --adv_sector_weight 0.3"
  "adv0.5__sectoronly:${SD} ${KEEP} --adv_sector_weight 0.5"
  "adv1.0__sectoronly:${SD} ${KEEP} --adv_sector_weight 1.0"
  # --- star-disjoint (stricter) ---
  "secdisjoint__stardisjoint:${SD}"
  "balance__stardisjoint:${SD} --balance_sector_age"
  "adv0.3__stardisjoint:${SD} --adv_sector_weight 0.3"
  "adv0.5__stardisjoint:${SD} --adv_sector_weight 0.5"
  "adv1.0__stardisjoint:${SD} --adv_sector_weight 1.0"
)

for entry in "${points[@]}"; do
  name="${entry%%:*}"; extra="${entry#*:}"
  outdir="${OUTBASE}/${name}"
  echo ""
  echo "================================================================"
  echo "[$(date '+%H:%M:%S')] SWEEP POINT: ${name}"
  echo "  extra flags: ${extra}"
  echo "  output: ${outdir}"
  echo "================================================================"
  python scripts/kfold_age_inference.py ${COMMON} --output_dir "${outdir}" ${extra} \
    || echo "  !! ${name} FAILED (continuing)"
done

echo ""
echo "================ SWEEP SUMMARY ================"
python3 - <<PY
import json, os, glob
base = "${OUTBASE}"
rows = []
for d in sorted(glob.glob(base + "/*/")):
    p = os.path.join(d, "kfold_metrics.json")
    if os.path.exists(p):
        m = json.load(open(p))
        rows.append((os.path.basename(d.rstrip("/")), m.get("n_samples"),
                     m.get("correlation"), m.get("mae_dex")))
print(f"{'config':22}{'N':>7}{'r':>9}{'MAE_dex':>9}")
for name, n, r, mae in rows:
    print(f"{name:22}{n:>7}{r:>9.4f}{mae:>9.4f}")
PY
