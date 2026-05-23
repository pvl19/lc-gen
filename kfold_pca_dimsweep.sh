#!/bin/bash
# PCA bottleneck-dim sweep (4 / 8 / 16) on the FINAL sendit/e50 latents.
# For each dim: a leave-one-cluster-out (LOCO) run + a matched random-10-fold
# baseline, all on ChronoFlow with the shared 11-fold grouping and clean
# ChronoFlow ages (--override_ages_from_csv). The latents were extracted with the
# locked-in pooling (uniform/equal_count/dt), so the kfold just consumes them.
#
# Goal: which PCA dim best trades capacity vs out-of-cluster overfitting.
#   pca{D}_random : random 10-fold (in-distribution reference, per dim)
#   pca{D}_loco   : leave-one-cluster-out (generalization, per dim)
# Baseline for context: gyro LOCO r=0.498 (model-independent — uses Prot, not latents;
# see final_model/parallel_fixed/e110/loocv/gyro/).

LAT=final_model/sendit/e50/metaAll/latents_pretrain.npz
BASE=final_model/sendit/e50/loocv_pcadim
COMMON="--load_latents ${LAT} --age_csv final_pretrain/metadata.csv --override_ages_from_csv \
  --pooling_mode multiscale --star_aggregation latent_max --use_metadata \
  --subset_col ref --subset_val ChronoFlow --subset_csv final_pretrain/metadata.csv \
  --loga_grid_size 1000 --seed 42 --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --encoder_type pca --training_stages joint --n_epochs 100"

for D in 4 8 16; do
  echo "############## PCA dim ${D} — random 10-fold ##############"
  python scripts/kfold_age_inference.py ${COMMON} --pca_dim ${D} --n_folds 10 \
    --output_dir ${BASE}/pca${D}_random || echo "  !! pca${D}_random FAILED"
  echo "############## PCA dim ${D} — LOCO ##############"
  python scripts/kfold_age_inference.py ${COMMON} --pca_dim ${D} --n_folds 11 --loocv_age \
    --output_dir ${BASE}/pca${D}_loco || echo "  !! pca${D}_loco FAILED"
done

echo ""
echo "================ PCA DIM SWEEP SUMMARY (sendit/e50) ================"
python3 - <<PY
import json, os
import pandas as pd, numpy as np
base = "${BASE}"
print(f"{'run':16}{'N':>7}{'r':>9}{'MAE':>9}")
for D in (4, 8, 16):
    for kind in ('random', 'loco'):
        d = f'pca{D}_{kind}'
        p = os.path.join(base, d, 'kfold_metrics.json')
        if os.path.exists(p):
            m = json.load(open(p))
            print(f"{d:16}{m['n_samples']:>7}{m['correlation']:>9.3f}{m['mae_dex']:>9.3f}")
        else:
            print(f"{d:16}{'(missing)':>7}")
print("\ncontext: gyro LOCO r=0.498 MAE=0.553 (model-independent baseline)")
print("\nPer-fold (age-group) MAE [dex] — LOCO runs:")
for D in (4, 8, 16):
    csv = os.path.join(base, f'pca{D}_loco', 'kfold_predictions.csv')
    if not os.path.exists(csv): continue
    df = pd.read_csv(csv)
    tcol = 'log10_true_age' if 'log10_true_age' in df.columns else [c for c in df.columns if 'true' in c.lower()][0]
    pcol = 'log10_pred_median' if 'log10_pred_median' in df.columns else [c for c in df.columns if 'pred' in c.lower()][0]
    df['ae'] = (df[pcol]-df[tcol]).abs(); df['age'] = (10**df[tcol]).round(0)
    g = df.groupby('fold').agg(lo=('age','min'), hi=('age','max'), n=('ae','size'), mae=('ae','mean')).sort_values('lo')
    print(f"  --- pca{D}_loco ---")
    for _,r in g.iterrows():
        rng = f"{r.lo:.0f}" if r.lo==r.hi else f"{r.lo:.0f}-{r.hi:.0f}"
        print(f"    {rng:>14} Myr  n={int(r.n):>4}  MAE={r.mae:.3f}")
PY
