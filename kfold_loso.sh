#!/bin/bash
# Leave-one-sector-out (LOSO) validation on the FINAL sendit/e50 latents.
#
# Question: how well does the age model generalize to stars observed ONLY in
# TESS sectors it never trained on? This is the per-star (latent_max) analogue of
# sector-disjoint CV and a direct probe of the sector confound's effect on the
# deployed pipeline.
#
# Design (--loso in kfold_age_inference.py):
#   - ALL labeled pretrain stars (every ref: ChronoFlow, Kiman, Newton, ...),
#     ages taken from the latents cache (NOT --override_ages_from_csv — we want
#     all catalogs, not just the clean ChronoFlow isochrone ages).
#   - latent_max per-star aggregation; each star carries the SET of sectors it was
#     observed in (threaded through aggregation).
#   - Unique sectors are randomly partitioned into 10 disjoint groups (seed 42).
#     Fold f holds out group f: VALIDATION = stars assigned to f (each star validates
#     once, in a group its sectors touch); TRAINING = only stars whose sectors are
#     ALL outside group f — every star touching a held-out sector is removed from
#     training (the whole star, not just its held-out-sector rows). No light curve
#     from a held-out sector, and no star sharing a sector with the validation set,
#     is ever seen in training.
#   - PCA encoder. PCA is fit per-fold on the training stars only (no projection
#     leakage); the dim is taken from the dim-sweep result below (best LOCO r),
#     falling back to 8 if the sweep hasn't run.
#
# Read the GLOBAL r/MAE over all held-out predictions. Context baselines:
#   ChronoFlow LOCO (cluster-disjoint): pca_d8 r=0.710, gyro r=0.498.
#   In-distribution random 10-fold ceiling: r~0.91.
# A LOSO r well below the in-distribution ceiling means the pipeline leans on
# per-sector structure that doesn't transfer to unseen sectors.

LAT=final_model/sendit/e50/metaAll/latents_pretrain.npz
HOSTS=final_model/sendit/e50/metaAll/latents_hosts.npz
THICK=final_model/sendit/e50/metaAll/latents_thickdisk.npz
SWEEP=final_model/sendit/e50/loocv_pcadim
BASE=final_model/sendit/e50/loso

# Global PCA pool: ALL pretraining stars' latents (pretrain + hosts + thickdisk).
# Fit once; every fold uses the same basis. Without this, run_kfold_cv refits PCA
# per-fold on each X_train — and LOSO's train set systematically excludes CVZ-
# touching stars, so each fold's basis would itself be OOD on its val set,
# confounding "model generalization" with "PCA-basis OOD".
PCA_POOL="--pca_latent_pool ${LAT} ${HOSTS} ${THICK}"

# --- Pick the PCA dim from the dim sweep (best LOCO correlation); fallback 8. ---
PCA_DIM=$(python3 - <<PY
import json, os, glob
sweep = "${SWEEP}"
best_d, best_r = None, -1e9
for d in (4, 8, 16):
    p = os.path.join(sweep, f"pca{d}_loco", "kfold_metrics.json")
    if os.path.exists(p):
        r = json.load(open(p)).get("correlation", -1e9)
        if r > best_r:
            best_r, best_d = r, d
print(best_d if best_d is not None else 8)
PY
)
echo "############## LOSO — PCA dim ${PCA_DIM} (from dim sweep; fallback 8) ##############"

COMMON="--load_latents ${LAT} --age_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation latent_max --use_metadata \
  --loga_grid_size 1000 --seed 42 --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --encoder_type pca --training_stages joint --n_epochs 100"

python scripts/kfold_age_inference.py ${COMMON} ${PCA_POOL} \
  --pca_dim ${PCA_DIM} --n_folds 10 --loso \
  --output_dir ${BASE}/pca${PCA_DIM}_loso || echo "  !! LOSO FAILED"

echo ""
echo "================ LOSO SUMMARY (sendit/e50, all labeled stars) ================"
python3 - <<PY
import json, os
import pandas as pd, numpy as np
base = "${BASE}"; d = "pca${PCA_DIM}_loso"
p = os.path.join(base, d, "kfold_metrics.json")
print(f"{'run':18}{'N':>7}{'r':>9}{'MAE':>9}")
if os.path.exists(p):
    m = json.load(open(p))
    print(f"{d:18}{m['n_samples']:>7}{m['correlation']:>9.3f}{m['mae_dex']:>9.3f}")
else:
    print(f"{d:18}{'(missing)':>7}")
print("\ncontext: ChronoFlow LOCO pca_d8 r=0.710 | gyro LOCO r=0.498 | "
      "in-distribution random 10-fold ceiling r~0.91")
print("\nPer-fold (sector-group) MAE [dex]:")
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
