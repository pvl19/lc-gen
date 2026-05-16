"""Partial-correlation + sector-probe tests for each latent variant.

For each variant directory, runs these probes on all-labeled-stars:

  (A) Age regression — three predictors, same 10-fold star-grouped split:
      - sector-only:  Ridge(one_hot_sector)             -> log_age
      - latent-only:  Ridge(latent)                     -> log_age
      - joint:        Ridge([latent | one_hot_sector])  -> log_age
      Per-star predictions = mean across that star's sectors.
      Reports r, MAE per predictor + (r_joint - r_sector) = "info latent adds over sector".

  (B) Partial correlation — does the latent predict the WITHIN-sector age residual?
      residual_i = log_age_i - mean(log_age | sector_i, train fold only)
      latent-only Ridge regresses on residual; r(pred_residual, true_residual) is the
      "physics beyond sector" signal.

  (C) Sector probe — multinomial logistic regression: latent -> sector_id.
      Star-grouped 80/20 train/test split. Top-1 and top-3 accuracy.
      High accuracy = latents partly encode sector identity.

Loads latents from final_model/parallel_fixed/<variant>/latents_{pretrain,hosts}.npz
and joins ages from final_pretrain/all_ages.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path('/Users/philvanlane/Documents/lc_ae')
AGE_CSV = ROOT / 'final_pretrain/all_ages.csv'

VARIANTS = sys.argv[1:] or [
    'e110',
    'e110_timeaware',
    'e110_no_seg',
    'e110_gapaware',
]


def load_variant(variant: str):
    parts = []
    for name in ('latents_pretrain.npz', 'latents_hosts.npz'):
        p = ROOT / f'final_model/parallel_fixed/{variant}/{name}'
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        key = 'latent_vectors' if 'latent_vectors' in z.files else 'latents'
        gid_key = 'gaia_ids' if 'gaia_ids' in z.files else 'tic_ids'
        parts.append({
            'X': z[key].astype(np.float32),
            'gid': z[gid_key].astype(np.int64),
            'sec': z['sectors'].astype(np.int64),
        })
    X = np.concatenate([p['X'] for p in parts], axis=0)
    gid = np.concatenate([p['gid'] for p in parts], axis=0)
    sec = np.concatenate([p['sec'] for p in parts], axis=0)
    return X, gid, sec


def main():
    ages = pd.read_csv(AGE_CSV)
    ages['log_age'] = np.log10(ages['age_Myr'].clip(lower=0.1))
    age_map = dict(zip(ages['GaiaDR3_ID'].astype(np.int64).values,
                       ages['log_age'].values))

    summary_rows = []
    for variant in VARIANTS:
        t0 = time.time()
        try:
            X, gid, sec = load_variant(variant)
        except FileNotFoundError as e:
            print(f'\n=== {variant}: SKIP ({e}) ===')
            continue
        print(f'\n=== {variant}: latents={X.shape}, unique stars={len(np.unique(gid))} ===')

        # Filter to labeled stars
        log_age_per_obs = np.array([age_map.get(int(g), np.nan) for g in gid])
        mask = np.isfinite(log_age_per_obs)
        X, gid, sec, y = X[mask], gid[mask], sec[mask], log_age_per_obs[mask]
        print(f'  labeled rows: {len(X)}, labeled stars: {len(np.unique(gid))}')

        # One-hot encode sectors observed across the whole variant
        sectors = np.sort(np.unique(sec))
        sec_idx = {s: i for i, s in enumerate(sectors)}
        S = np.zeros((len(X), len(sectors)), dtype=np.float32)
        S[np.arange(len(X)), [sec_idx[s] for s in sec]] = 1.0

        # Star-level stratification on per-star mean log_age
        star_ages = pd.Series(y).groupby(gid).mean()
        star_ids = star_ages.index.values
        star_y = star_ages.values
        bins = np.linspace(star_y.min(), star_y.max() + 1e-6, 21)
        star_bin = np.clip(np.digitize(star_y, bins) - 1, 0, 19)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        pred_sec   = np.full(len(y), np.nan)
        pred_lat   = np.full(len(y), np.nan)
        pred_joint = np.full(len(y), np.nan)
        pred_resid = np.full(len(y), np.nan)
        true_resid = np.full(len(y), np.nan)

        for fold, (tr_star_idx, te_star_idx) in enumerate(skf.split(star_ids, star_bin)):
            tr_stars = set(int(s) for s in star_ids[tr_star_idx])
            te_stars = set(int(s) for s in star_ids[te_star_idx])
            tr = np.array([int(g) in tr_stars for g in gid])
            te = np.array([int(g) in te_stars for g in gid])

            # Normalize latents (fit on train fold)
            scaler = StandardScaler(with_mean=True, with_std=True).fit(X[tr])
            Xn_tr = scaler.transform(X[tr])
            Xn_te = scaler.transform(X[te])

            # (A1) sector-only
            r_sec = Ridge(alpha=1.0).fit(S[tr], y[tr])
            pred_sec[te] = r_sec.predict(S[te])
            # (A2) latent-only
            r_lat = Ridge(alpha=1.0).fit(Xn_tr, y[tr])
            pred_lat[te] = r_lat.predict(Xn_te)
            # (A3) joint
            J_tr = np.concatenate([Xn_tr, S[tr]], axis=1)
            J_te = np.concatenate([Xn_te, S[te]], axis=1)
            r_joint = Ridge(alpha=1.0).fit(J_tr, y[tr])
            pred_joint[te] = r_joint.predict(J_te)

            # (B) partial: residual = y - sector_mean(y, train fold)
            sec_mean_train = {}
            for s in sectors:
                m = tr & (sec == s)
                sec_mean_train[s] = y[m].mean() if m.any() else y[tr].mean()
            te_sec_pred = np.array([sec_mean_train[s] for s in sec[te]])
            true_resid[te] = y[te] - te_sec_pred
            tr_sec_pred = np.array([sec_mean_train[s] for s in sec[tr]])
            resid_train = y[tr] - tr_sec_pred
            r_resid = Ridge(alpha=1.0).fit(Xn_tr, resid_train)
            pred_resid[te] = r_resid.predict(Xn_te)

        # Aggregate per star: mean across sectors
        def per_star(arr):
            return pd.Series(arr).groupby(gid).mean().reindex(star_ids).values
        true_star = per_star(y)
        sec_star  = per_star(pred_sec)
        lat_star  = per_star(pred_lat)
        join_star = per_star(pred_joint)
        # For residual, aggregate true and pred at star level then correlate
        true_res_star = per_star(true_resid)
        pred_res_star = per_star(pred_resid)

        def report(name, pred, target):
            ok = np.isfinite(pred) & np.isfinite(target)
            r, _ = pearsonr(pred[ok], target[ok])
            mae = np.mean(np.abs(pred[ok] - target[ok]))
            return r, mae

        r_s, m_s = report('sector', sec_star, true_star)
        r_l, m_l = report('latent', lat_star, true_star)
        r_j, m_j = report('joint',  join_star, true_star)
        r_r, m_r = report('residual', pred_res_star, true_res_star)

        # (C) sector probe — logistic regression latent -> sector_id, star-grouped 80/20
        tr_idx, te_idx = train_test_split(np.arange(len(star_ids)), test_size=0.2,
                                          random_state=42, stratify=star_bin)
        tr_stars = set(int(s) for s in star_ids[tr_idx])
        te_stars = set(int(s) for s in star_ids[te_idx])
        tr_mask = np.array([int(g) in tr_stars for g in gid])
        te_mask = np.array([int(g) in te_stars for g in gid])
        scaler2 = StandardScaler().fit(X[tr_mask])
        Xtr = scaler2.transform(X[tr_mask])
        Xte = scaler2.transform(X[te_mask])
        ytr = sec[tr_mask]
        yte = sec[te_mask]
        # Multinomial logistic regression. Limit iter for speed.
        clf = LogisticRegression(max_iter=200, solver='lbfgs', C=1.0, n_jobs=-1)
        clf.fit(Xtr, ytr)
        # Top-1 / top-3 accuracy
        probs = clf.predict_proba(Xte)
        classes = clf.classes_
        top1 = classes[np.argmax(probs, axis=1)]
        acc1 = float(np.mean(top1 == yte))
        top3 = classes[np.argsort(-probs, axis=1)[:, :3]]
        acc3 = float(np.mean(np.any(top3 == yte[:, None], axis=1)))
        # Chance baseline = max-class frequency in train
        unique, counts = np.unique(ytr, return_counts=True)
        chance = float(counts.max() / counts.sum())

        dur = time.time() - t0
        print(f'  (A) age-from-X (star-level, 10-fold):')
        print(f'        sector-only        r={r_s:.4f}  MAE={m_s:.4f} dex')
        print(f'        latent-only        r={r_l:.4f}  MAE={m_l:.4f} dex')
        print(f'        joint              r={r_j:.4f}  MAE={m_j:.4f} dex')
        print(f'        joint - sector     Δr={r_j-r_s:+.4f}  ΔMAE={m_j-m_s:+.4f}')
        print(f'  (B) partial: latent on within-sector residual')
        print(f'        r={r_r:.4f}  MAE={m_r:.4f} dex   (0 = no info beyond sector)')
        print(f'  (C) sector probe (latent -> sector_id, 80/20 star-grouped):')
        print(f'        top-1={acc1:.4f}   top-3={acc3:.4f}   chance={chance:.4f}')
        print(f'  ({dur:.1f}s)')
        summary_rows.append({
            'variant': variant,
            'n_stars': len(star_ids),
            'r_sector_only': r_s, 'mae_sector_only': m_s,
            'r_latent_only': r_l, 'mae_latent_only': m_l,
            'r_joint': r_j, 'mae_joint': m_j,
            'dr_joint_over_sector': r_j - r_s,
            'r_residual': r_r, 'mae_residual': m_r,
            'probe_top1': acc1, 'probe_top3': acc3, 'probe_chance': chance,
        })

    print('\n\nSUMMARY')
    print('=' * 100)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 50)
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
