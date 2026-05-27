# Age inference results — `sendit/e50` model

Canonical root for all age-inference test results on the final `sendit/e50`
pretrained autoencoder. Notebooks should point here as a single location;
every subfolder follows the layout below and contains the standard
`kfold_metrics.json`, `kfold_predictions.csv`, `kfold_results.png`,
`kfold_models.pt`, `training_curves.json`.

All latent-flow runs use:
- `--encoder_type pca`, `--training_stages joint`, `--n_epochs 100`
- `--flow_transforms 6 --flow_hidden_dims 64 64`
- Global PCA basis from `shared/global_pca_d16.npz` (fit ONCE on all ~69K
  per-sector latents across pretrain + hosts + thickdisk; identical at dims 4/8/16
  via truncation — see `--pca_cache` in `scripts/kfold_age_inference.py`).
- All gyro baselines use the standard `kfold_gyro_baseline.py` (per-star,
  Prot + colour, no light-curve features).

Ages are in **log₁₀(age / Myr)** in every output CSV's `log10_true_age` /
`log10_pred_median` columns. MAE is reported in dex (= |Δ log₁₀(age)|).

---

## Headline summary

PCA dim 4 throughout (best LOCO from the dim sweep).

### ChronoFlow population (N = 2,470 cluster stars, isochrone ages)

| test | location | r | MAE (dex) |
|---|---|---|---|
| **In-distribution ceiling** (random 10-fold, dim 8) | `chronoflow/random_latent_ceiling/pca8/` | 0.828 | 0.278 |
| **Random per-LC star-disjoint** (latent, predict_mean) | `chronoflow/random_latent/pca4/` | **0.847** | 0.282 |
| **Random per-star** (gyro baseline) | `chronoflow/random_gyro/` | 0.503 | 0.536 |
| **LOCO dim 4** (cluster-disjoint, winner) | `chronoflow/loco/pca4/` | **0.709** | 0.457 |
| LOCO dim 8 | `chronoflow/loco/pca8/` | 0.668 | 0.511 |
| LOCO dim 16 | `chronoflow/loco/pca16/` | 0.660 | 0.473 |
| **LOCO gyro reference** (from e110 era) | `chronoflow/loco_gyro_reference/` | 0.498 | 0.553 |
| **LOSO** (sector-disjoint, balanced LPT) | `chronoflow/loso/pca4/` | **0.732** | 0.473 |

### All labeled pretrain stars (N = 9,221, mixed catalogs — gyro-derived ages dominate the non-cluster subset)

| test | location | r | MAE (dex) |
|---|---|---|---|
| LOSO strict (training excludes touching stars) | `all_pretrain/loso_strict/pca4/` | 0.454 | 0.542 |
| LOSO relaxed (training = all non-val stars) | `all_pretrain/loso_relaxed/pca4/` | 0.465 | 0.511 |

### Pretrain + hosts (N = 9,413 labeled — +192 cluster-member hosts)

| test | location | r | MAE (dex) |
|---|---|---|---|
| Random per-LC star-disjoint (latent, predict_mean) | `all_pretrain_plus_hosts/random_latent/pca4/` | 0.623 | 0.420 |
| Random per-star (gyro, N=3,023 with Prot) | `all_pretrain_plus_hosts/random_gyro/` | 0.562 | 0.502 |
| LOSO (sector-disjoint, balanced LPT) | `all_pretrain_plus_hosts/loso/pca4/` | 0.455 | 0.590 |

### Archive (prior per-fold PCA — kept for comparison)

These were superseded by the global PCA cache. Per-fold PCA introduced
fold-to-fold basis noise that artificially depressed the LOCO numbers.

| test | location |
|---|---|
| dim sweep (per-fold PCA): pca4/8/16 LOCO + pca8_random | `_archive/perfold_pca_loocv/` |
| LOSO (per-fold PCA): r=0.471 MAE=0.513 | `_archive/perfold_pca_loso/pca4_loso/` |

---

## Folder map

```
final_model/sendit/e50/age_inference/
├── README.md                          ← this file
├── shared/
│   └── global_pca_d16.npz             ← fit once on ~69K latents (16 components); every run loads + truncates
│
├── chronoflow/                        ← N=2,470 ChronoFlow stars, isochrone ages
│   │                                    --override_ages_from_csv --subset_val ChronoFlow
│   ├── loco/{pca4,pca8,pca16}/        ← leave-one-cluster-out, dim sweep (winner: pca4)
│   ├── random_latent_ceiling/pca8/    ← in-distribution ceiling: random 10-fold, latent_max, dim 8
│   ├── loso/pca4/                     ← leave-one-sector-out, balanced-LPT folds, latent_max
│   ├── random_latent/pca4/            ← random star-disjoint kfold, per-LC training (predict_mean)
│   ├── random_gyro/                   ← random star-disjoint kfold, gyro baseline (Prot + colour)
│   └── loco_gyro_reference/           ← copied from final_model/parallel_fixed/e110/loocv/gyro/
│
├── all_pretrain/                      ← N=9,221 all labeled pretrain stars
│   ├── loso_strict/pca4/              ← --loso (training excludes stars touching held-out sectors)
│   └── loso_relaxed/pca4/             ← --loso --loso_relaxed_train (training = all non-val stars)
│
├── all_pretrain_plus_hosts/           ← N=9,413: pretrain + ~192 cluster-member exoplanet hosts
│   ├── loso/pca4/                     ← --loso, multi-cache --load_latents (pretrain + hosts)
│   ├── random_latent/pca4/            ← random per-LC star-disjoint (predict_mean), multi-cache
│   └── random_gyro/                   ← random kfold, gyro baseline (pretrain only — hosts have no Prot)
│
└── _archive/                          ← prior per-fold-PCA runs, kept for comparison
    ├── perfold_pca_loocv/             ← prior dim sweep
    └── perfold_pca_loso/              ← prior LOSO (N=9,221)
```

---

## Reproducibility — which wrapper writes here

| wrapper | output destination(s) |
|---|---|
| `kfold_pca_dimsweep.sh` | `chronoflow/loco/{pca4,pca8,pca16}/`, `chronoflow/random_latent_ceiling/pca8/` (also creates `shared/global_pca_d16.npz` if missing) |
| `kfold_loso_chronoflow.sh` | `chronoflow/loso/pca${PCA_DIM}/` |
| `kfold_random_chronoflow.sh` | `chronoflow/random_latent/pca${PCA_DIM}/`, `chronoflow/random_gyro/` |
| `kfold_loso.sh` | `all_pretrain/loso_strict/pca${PCA_DIM}/` |
| `kfold_loso_relaxed.sh` | `all_pretrain/loso_relaxed/pca${PCA_DIM}/` |
| `kfold_allages_tests.sh` | `all_pretrain_plus_hosts/{random_latent,random_gyro,loso}/...` |

Each wrapper reads `PCA_DIM` from the dim sweep (best LOCO r). All wrappers
share the same global PCA cache; the first wrapper to run after the cache
file is deleted will rebuild it (~30 s).

---

## Shared input data (kept outside this folder, referenced by path)

```
final_model/sendit/e50/best_model.pt                       ← the trained autoencoder
final_model/sendit/e50/metaAll/latents_pretrain.npz        ← per-sector latents (pretrain H5)
final_model/sendit/e50/metaAll/latents_hosts.npz           ← per-sector latents (hosts H5)
final_model/sendit/e50/metaAll/latents_thickdisk.npz       ← per-sector latents (thickdisk H5)
final_pretrain/metadata.csv                                ← per-(star,ref) ages + photometry + Prot
final_pretrain/host_all_metadata.csv                       ← host photometry (no ages, no Prot)
```

Latents weren't moved into this folder to avoid duplicating ~435 MB. They're
inputs; this folder is for outputs.

---

## Notes on interpretation

- **LOCO–LOSO gap is essentially zero on matched populations.** ChronoFlow LOCO
  (0.709) ≈ ChronoFlow LOSO (0.732). The encoder is *not* sector-confounded
  for cluster stars. The all-ages LOSO drop (0.455) comes overwhelmingly from
  the noisier gyro-derived ages of non-cluster M-dwarf rotation-survey stars,
  not from sector pathology.
- **Latent ≥ gyro on every matched-population scheme.** Margin is largest at
  random (Δr ≈ +0.34 on ChronoFlow), smallest at all-ages random (Δr ≈ +0.06,
  inflated by the gyro-circularity in non-cluster label generation).
- **MAE columns are in dex** (= |log₁₀(pred_age) − log₁₀(true_age)|). A MAE of
  ~0.45 dex corresponds to ~3× factor errors in linear age; 0.28 dex ~ 2× factor.
