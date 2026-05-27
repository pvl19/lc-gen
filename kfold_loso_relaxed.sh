#!/bin/bash
# Relaxed-train LOSO on the FINAL sendit/e50 latents.
#
# This is the same per-star (latent_max) leave-one-sector-out validation as
# kfold_loso.sh, but with the training-set rule RELAXED: stars that touch the
# held-out sector group are KEPT in training (training = all stars NOT in val),
# rather than the strict version's "remove the whole touching star". Same val
# partition either way — it's still stars-assigned-to-fold-f based on the
# greedy LPT sector-group assignment.
#
# Why this is a defensible test at PCA dim 4: the bottleneck is too low and the
# NSF flow too smooth to memorize per-star age examples. Training and val will
# include similar (latent_max → age) regions, but the model can't store
# point-wise lookups; it has to learn a smooth function over a 4-D space. So
# the relaxation tests "sector-aware validation, full training context" without
# leaking memorized stellar ages.
#
# Comparison points (sendit/e50, PCA dim 4, global PCA cache, ages from cache):
#   strict LOSO   (age_inference/all_pretrain/loso_strict/pca4/): r=0.454 MAE=0.542 N=9221
#   in-distribution ceiling (ChronoFlow N=2470, random 10-fold):  r=0.828
# If relaxed-LOSO is close to the strict number → the gap is the val-side sector
# partition itself, not the train-side data removal. If it's close to the ceiling
# → the strict train-side exclusion was the dominant cost.

LAT=final_model/sendit/e50/metaAll/latents_pretrain.npz
HOSTS=final_model/sendit/e50/metaAll/latents_hosts.npz
THICK=final_model/sendit/e50/metaAll/latents_thickdisk.npz
AGE_ROOT=final_model/sendit/e50/age_inference
SWEEP_LOCO=${AGE_ROOT}/chronoflow/loco
BASE=${AGE_ROOT}/all_pretrain/loso_relaxed
STRICT_REF=${AGE_ROOT}/all_pretrain/loso_strict

PCA_CACHE=${AGE_ROOT}/shared/global_pca_d16.npz
PCA_POOL="--pca_latent_pool ${LAT} ${HOSTS} ${THICK} --pca_cache ${PCA_CACHE} --pca_cache_max_dim 16"

# Pick the PCA dim from the dim sweep (best LOCO r); fallback 8.
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
echo "############## RELAXED LOSO — PCA dim ${PCA_DIM} ##############"

COMMON="--load_latents ${LAT} --age_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation latent_max --use_metadata \
  --loga_grid_size 1000 --seed 42 --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --encoder_type pca --training_stages joint --n_epochs 100"

python scripts/kfold_age_inference.py ${COMMON} ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 10 --loso --loso_relaxed_train \
  --output_dir ${BASE}/pca${PCA_DIM} || echo "  !! RELAXED LOSO FAILED"

echo ""
echo "================ RELAXED LOSO SUMMARY (sendit/e50, all labeled stars) ================"
python3 - <<PY
import json, os
import pandas as pd, numpy as np
base = "${BASE}"; d = "pca${PCA_DIM}"
p = os.path.join(base, d, "kfold_metrics.json")
print(f"{'run':22}{'N':>7}{'r':>9}{'MAE':>9}")
if os.path.exists(p):
    m = json.load(open(p))
    print(f"{d+' (relaxed)':22}{m['n_samples']:>7}{m['correlation']:>9.3f}{m['mae_dex']:>9.3f}")
# Also report the strict number for direct comparison
ps = os.path.join("${STRICT_REF}", d, "kfold_metrics.json")
if os.path.exists(ps):
    m = json.load(open(ps))
    print(f"{d+' (strict, ref)':22}{m['n_samples']:>7}{m['correlation']:>9.3f}{m['mae_dex']:>9.3f}")
print("\nPer-fold (sector-group) MAE [dex] — relaxed:")
csv = os.path.join(base, d, "kfold_predictions.csv")
if os.path.exists(csv):
    df = pd.read_csv(csv)
    tcol = 'log10_true_age' if 'log10_true_age' in df.columns else [c for c in df.columns if 'true' in c.lower()][0]
    pcol = 'log10_pred_median' if 'log10_pred_median' in df.columns else [c for c in df.columns if 'pred' in c.lower()][0]
    df['ae'] = (df[pcol]-df[tcol]).abs()
    g = df.groupby('fold').agg(n=('ae','size'), mae=('ae','mean'),
                               r=(pcol, lambda s: np.corrcoef(s, df.loc[s.index, tcol])[0,1]
                                  if len(s) > 2 and df.loc[s.index, tcol].std() > 0 else np.nan))
    for f, row in g.iterrows():
        print(f"    fold {int(f):>2}  n={int(row.n):>4}  MAE={row.mae:.3f}  r={row.r:.3f}")
PY
