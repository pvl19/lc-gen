#!/bin/bash
# LOSO restricted to the ChronoFlow subset (clean isochrone ages, same N=2,470
# population as the LOCO dim sweep). Same balanced-LPT + greedy-val LOSO
# mechanics as kfold_loso.sh, just with --subset_col/--subset_val/--override.
#
# Purpose: disentangle the LOCO→LOSO global-r gap into (a) population/label-
# quality vs (b) sector-vs-cluster generalization-direction effects. Reference
# points (sendit/e50, PCA dim 4, global PCA cache, balanced folds):
#   LOCO (ChronoFlow, cluster-disjoint):   r=0.709  MAE=0.457  N=2470
#   LOSO (ALL labeled, sector-disjoint):   r=0.454  MAE=0.542  N=9221
#   LOSO (ChronoFlow, sector-disjoint):    <this run>
#
# Interpretation:
#   ≈0.45  → population/labels matter little; sector-axis generalization is the
#            dominant gap.
#   ~0.55-0.65 → both effects real; mixed-catalog ages add ~0.1 to the gap.
#   ~0.70  → it's almost entirely population/labels; LOCO ≈ LOSO when matched.

LAT=final_model/sendit/e50/metaAll/latents_pretrain.npz
HOSTS=final_model/sendit/e50/metaAll/latents_hosts.npz
THICK=final_model/sendit/e50/metaAll/latents_thickdisk.npz
AGE_ROOT=final_model/sendit/e50/age_inference
SWEEP_LOCO=${AGE_ROOT}/chronoflow/loco
BASE=${AGE_ROOT}/chronoflow/loso

# Same global PCA cache as the dim sweep + full-set LOSO — the basis is
# population-agnostic, so the ChronoFlow restriction only narrows the kfold
# sample, not the projection.
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
echo "############## LOSO (ChronoFlow only) — PCA dim ${PCA_DIM} ##############"

# Same training hyperparams + LOSO mechanics as kfold_loso.sh; the differences
# are --subset_col/--subset_val/--subset_csv (restrict to ChronoFlow stars) and
# --override_ages_from_csv (use the clean isochrone ages from metadata.csv,
# not the blended cache ages).
COMMON="--load_latents ${LAT} --age_csv final_pretrain/metadata.csv --override_ages_from_csv \
  --subset_col ref --subset_val ChronoFlow --subset_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation latent_max --use_metadata \
  --loga_grid_size 1000 --seed 42 --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --encoder_type pca --training_stages joint --n_epochs 100"

python scripts/kfold_age_inference.py ${COMMON} ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 10 --loso \
  --output_dir ${BASE}/pca${PCA_DIM} || echo "  !! LOSO-ChronoFlow FAILED"

echo ""
echo "================ LOSO-ChronoFlow SUMMARY ================"
python3 - <<PY
import json, os
import pandas as pd, numpy as np
runs = [
    ("LOCO (ChronoFlow)",     "${SWEEP_LOCO}/pca${PCA_DIM}/kfold_metrics.json"),
    ("LOSO (ALL labeled)",    "${AGE_ROOT}/all_pretrain/loso_strict/pca${PCA_DIM}/kfold_metrics.json"),
    ("LOSO (ChronoFlow)",     "${BASE}/pca${PCA_DIM}/kfold_metrics.json"),
]
print(f"{'run':>22}{'N':>7}{'r':>9}{'MAE':>9}")
for name, p in runs:
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"{name:>22}{m['n_samples']:>7}{m['correlation']:>9.3f}{m['mae_dex']:>9.3f}")
    else:
        print(f"{name:>22}{'(missing)':>7}")
print("\nPer-fold (sector-group) MAE/r [LOSO-ChronoFlow]:")
csv = "${BASE}/pca${PCA_DIM}/kfold_predictions.csv"
if os.path.exists(csv):
    df = pd.read_csv(csv)
    tcol = 'log10_true_age' if 'log10_true_age' in df.columns else [c for c in df.columns if 'true' in c.lower()][0]
    pcol = 'log10_pred_median' if 'log10_pred_median' in df.columns else [c for c in df.columns if 'pred' in c.lower()][0]
    df['ae'] = (df[pcol]-df[tcol]).abs()
    df['age_myr'] = (10**df[tcol]).round(0)
    g = df.groupby('fold').agg(n=('ae','size'),
                               mae=('ae','mean'),
                               r=(pcol, lambda s: np.corrcoef(s, df.loc[s.index, tcol])[0,1]
                                  if len(s) > 2 and df.loc[s.index, tcol].std() > 0 else np.nan),
                               ages=('age_myr', lambda s: ", ".join(str(int(a)) for a in
                                     pd.Series(s).value_counts().head(3).index)))
    for f, row in g.iterrows():
        print(f"  fold {int(f):>2}  n={int(row.n):>4}  MAE={row.mae:.3f}  r={row.r:+.3f}  top-ages: {row.ages}")
PY
