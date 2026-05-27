#!/bin/bash
# PCA bottleneck-dim sweep (4 / 8 / 16) on the FINAL sendit/e50 latents.
# For each dim: a leave-one-cluster-out (LOCO) run + a matched random-10-fold
# baseline, all on ChronoFlow with the shared 11-fold grouping and clean
# ChronoFlow ages (--override_ages_from_csv). The latents were extracted with the
# locked-in pooling (uniform/equal_count/dt), so the kfold just consumes them.
#
# Goal: which PCA dim best trades capacity vs out-of-cluster overfitting.
#   pca{4,8,16}_loco : leave-one-cluster-out (generalization) — this picks the dim
#   pca8_random      : ONE random 10-fold = in-distribution ceiling / memorization-gap
#                      reference (per-dim random is redundant — in-distribution all dims
#                      do ~equally, so one suffices).
# Baseline for context: gyro LOCO r=0.498 (model-independent — uses Prot, not latents;
# see final_model/parallel_fixed/e110/loocv/gyro/).

LAT=final_model/sendit/e50/metaAll/latents_pretrain.npz
HOSTS=final_model/sendit/e50/metaAll/latents_hosts.npz
THICK=final_model/sendit/e50/metaAll/latents_thickdisk.npz
# Canonical age-inference results root. All test outputs land here so notebooks
# can point at a single location. See age_inference/README.md for the map.
AGE_ROOT=final_model/sendit/e50/age_inference
BASE_LOCO=${AGE_ROOT}/chronoflow/loco
BASE_CEILING=${AGE_ROOT}/chronoflow/random_latent_ceiling

# Global PCA: fit ONCE on the full pool (pretrain + hosts + thickdisk) at dim 16,
# saved as a single artifact. Every run below loads + truncates from that file —
# numerically identical basis at dims 4/8/16, no per-fold or per-run refit.
# The first run below creates the cache; subsequent runs (and kfold_loso.sh) load.
PCA_CACHE=${AGE_ROOT}/shared/global_pca_d16.npz
PCA_POOL_OR_CACHE="--pca_latent_pool ${LAT} ${HOSTS} ${THICK} \
  --pca_cache ${PCA_CACHE} --pca_cache_max_dim 16"

COMMON="--load_latents ${LAT} --age_csv final_pretrain/metadata.csv --override_ages_from_csv \
  --pooling_mode multiscale --star_aggregation latent_max --use_metadata \
  --subset_col ref --subset_val ChronoFlow --subset_csv final_pretrain/metadata.csv \
  --loga_grid_size 1000 --seed 42 --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --encoder_type pca --training_stages joint --n_epochs 100 ${PCA_POOL_OR_CACHE}"

# One in-distribution ceiling reference (dim 8, random 10-fold).
echo "############## PCA dim 8 — random 10-fold (in-distribution ceiling) ##############"
python scripts/kfold_age_inference.py ${COMMON} --pca_dim 8 --n_folds 10 \
  --output_dir ${BASE_CEILING}/pca8 || echo "  !! pca8_random FAILED"

# Dim comparison: LOCO for each dim (this is what picks the dim).
for D in 4 8 16; do
  echo "############## PCA dim ${D} — LOCO ##############"
  python scripts/kfold_age_inference.py ${COMMON} --pca_dim ${D} --n_folds 11 --loocv_age \
    --output_dir ${BASE_LOCO}/pca${D} || echo "  !! pca${D}_loco FAILED"
done

echo ""
echo "================ PCA DIM SWEEP SUMMARY (sendit/e50) ================"
python3 - <<PY
import json, os
import pandas as pd, numpy as np
loco = "${BASE_LOCO}"; ceil = "${BASE_CEILING}"
print(f"{'run':18}{'N':>7}{'r':>9}{'MAE':>9}")
entries = [('random_ceiling/pca8', os.path.join(ceil, 'pca8', 'kfold_metrics.json'))]
for d in (4, 8, 16):
    entries.append((f'loco/pca{d}', os.path.join(loco, f'pca{d}', 'kfold_metrics.json')))
for name, p in entries:
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"{name:18}{m['n_samples']:>7}{m['correlation']:>9.3f}{m['mae_dex']:>9.3f}")
    else:
        print(f"{name:18}{'(missing)':>7}")
print("\ncontext: gyro LOCO r=0.498 MAE=0.553 (model-independent baseline)")
print("\nPer-fold (age-group) MAE [dex] — LOCO runs:")
for D in (4, 8, 16):
    csv = os.path.join(loco, f'pca{D}', 'kfold_predictions.csv')
    if not os.path.exists(csv): continue
    df = pd.read_csv(csv)
    tcol = 'log10_true_age' if 'log10_true_age' in df.columns else [c for c in df.columns if 'true' in c.lower()][0]
    pcol = 'log10_pred_median' if 'log10_pred_median' in df.columns else [c for c in df.columns if 'pred' in c.lower()][0]
    df['ae'] = (df[pcol]-df[tcol]).abs(); df['age'] = (10**df[tcol]).round(0)
    g = df.groupby('fold').agg(lo=('age','min'), hi=('age','max'), n=('ae','size'), mae=('ae','mean')).sort_values('lo')
    print(f"  --- loco/pca{D} ---")
    for _,r in g.iterrows():
        rng = f"{r.lo:.0f}" if r.lo==r.hi else f"{r.lo:.0f}-{r.hi:.0f}"
        print(f"    {rng:>14} Myr  n={int(r.n):>4}  MAE={r.mae:.3f}")
PY
