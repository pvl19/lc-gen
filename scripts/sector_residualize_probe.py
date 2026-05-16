"""Test 1 — sector-residualize latents and re-probe.

For each variant: in-fold residualization (no leakage):
  1. For each training fold, compute sector_mean(latent | training set, sector_s)
  2. Subtract from BOTH train and test rows: latent_clean = latent - sector_mean
  3. Refit Ridge(latent_clean -> log_age) and report r_latent_clean.
  4. Refit logistic(latent_clean -> sector_id) — accuracy should drop substantially
     if residualization actually scrubbed sector signal.

If r_latent_clean stays high → physics signal lives in deviations from sector means
(latents are usable, just need sector-mean removed).
If r_latent_clean collapses → latents mostly encoded sector-as-cluster-prior, deeper fix needed.

Usage: python /tmp/sector_residualize_probe.py [variant ...]
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


def residualize(X_tr, sec_tr, X_te, sec_te):
    """In-fold sector-mean residualization (no leakage)."""
    # Per-sector means on the training fold only
    sec_means = {}
    overall_mean = X_tr.mean(axis=0)
    for s in np.unique(sec_tr):
        m = sec_tr == s
        if m.any():
            sec_means[int(s)] = X_tr[m].mean(axis=0)
    def apply(X, sec):
        out = X.copy()
        for i, s in enumerate(sec):
            out[i] -= sec_means.get(int(s), overall_mean)
        return out
    return apply(X_tr, sec_tr), apply(X_te, sec_te)


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
        except FileNotFoundError:
            print(f'=== {variant}: SKIP ===')
            continue
        print(f'\n=== {variant}: latents={X.shape}, unique stars={len(np.unique(gid))} ===',
              flush=True)
        log_age_per_obs = np.array([age_map.get(int(g), np.nan) for g in gid])
        mask = np.isfinite(log_age_per_obs)
        X, gid, sec, y = X[mask], gid[mask], sec[mask], log_age_per_obs[mask]

        # Star-level stratification
        star_ages = pd.Series(y).groupby(gid).mean()
        star_ids = star_ages.index.values
        star_y = star_ages.values
        bins = np.linspace(star_y.min(), star_y.max() + 1e-6, 21)
        star_bin = np.clip(np.digitize(star_y, bins) - 1, 0, 19)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        pred_raw   = np.full(len(y), np.nan)
        pred_clean = np.full(len(y), np.nan)

        for fold, (tr_star_idx, te_star_idx) in enumerate(skf.split(star_ids, star_bin)):
            tr_stars = set(int(s) for s in star_ids[tr_star_idx])
            te_stars = set(int(s) for s in star_ids[te_star_idx])
            tr = np.array([int(g) in tr_stars for g in gid])
            te = np.array([int(g) in te_stars for g in gid])

            # Raw latents (baseline)
            sc_raw = StandardScaler().fit(X[tr])
            Xr_tr = sc_raw.transform(X[tr])
            Xr_te = sc_raw.transform(X[te])
            r_raw = Ridge(alpha=1.0).fit(Xr_tr, y[tr])
            pred_raw[te] = r_raw.predict(Xr_te)

            # Residualized latents
            Xc_tr_raw, Xc_te_raw = residualize(X[tr], sec[tr], X[te], sec[te])
            sc_clean = StandardScaler().fit(Xc_tr_raw)
            Xc_tr = sc_clean.transform(Xc_tr_raw)
            Xc_te = sc_clean.transform(Xc_te_raw)
            r_clean = Ridge(alpha=1.0).fit(Xc_tr, y[tr])
            pred_clean[te] = r_clean.predict(Xc_te)

        # Aggregate per star
        def per_star(arr):
            return pd.Series(arr).groupby(gid).mean().reindex(star_ids).values
        sr_raw   = per_star(pred_raw)
        sr_clean = per_star(pred_clean)
        true_star = star_y

        def metric(p):
            ok = np.isfinite(p)
            r, _ = pearsonr(p[ok], true_star[ok])
            mae = np.mean(np.abs(p[ok] - true_star[ok]))
            return r, mae

        r_raw_, m_raw_     = metric(sr_raw)
        r_clean_, m_clean_ = metric(sr_clean)

        # Sector probe on raw vs residualized latents (80/20 star-grouped)
        tr_idx, te_idx = train_test_split(np.arange(len(star_ids)), test_size=0.2,
                                          random_state=42, stratify=star_bin)
        tr_stars = set(int(s) for s in star_ids[tr_idx])
        te_stars = set(int(s) for s in star_ids[te_idx])
        tr_mask = np.array([int(g) in tr_stars for g in gid])
        te_mask = np.array([int(g) in te_stars for g in gid])

        def sector_probe(X_tr, X_te):
            sc = StandardScaler().fit(X_tr)
            Xtr_n = sc.transform(X_tr)
            Xte_n = sc.transform(X_te)
            clf = LogisticRegression(max_iter=200, solver='lbfgs', C=1.0)
            clf.fit(Xtr_n, sec[tr_mask])
            probs = clf.predict_proba(Xte_n)
            classes = clf.classes_
            top1 = classes[np.argmax(probs, axis=1)]
            acc1 = float(np.mean(top1 == sec[te_mask]))
            top3 = classes[np.argsort(-probs, axis=1)[:, :3]]
            acc3 = float(np.mean(np.any(top3 == sec[te_mask][:, None], axis=1)))
            return acc1, acc3

        # Raw probe
        a1_raw, a3_raw = sector_probe(X[tr_mask], X[te_mask])
        # Residualized probe (use in-fold means computed from tr_mask training set)
        Xc_tr_raw, Xc_te_raw = residualize(X[tr_mask], sec[tr_mask],
                                            X[te_mask], sec[te_mask])
        a1_cl, a3_cl = sector_probe(Xc_tr_raw, Xc_te_raw)

        dur = time.time() - t0
        print(f'  age regression (Ridge -> log_age, 10-fold star-grouped):', flush=True)
        print(f'      RAW latent       r={r_raw_:.4f}  MAE={m_raw_:.4f}')
        print(f'      RESIDUALIZED     r={r_clean_:.4f}  MAE={m_clean_:.4f}')
        print(f'      Δr = {r_clean_-r_raw_:+.4f}   ΔMAE = {m_clean_-m_raw_:+.4f}')
        print(f'  sector probe (logreg -> sector_id, 80/20 star-grouped):', flush=True)
        print(f'      RAW latent       top1={a1_raw:.4f}  top3={a3_raw:.4f}')
        print(f'      RESIDUALIZED     top1={a1_cl:.4f}   top3={a3_cl:.4f}')
        print(f'      Δtop1 = {a1_cl-a1_raw:+.4f}')
        print(f'  ({dur:.1f}s)', flush=True)
        summary_rows.append({
            'variant': variant,
            'r_raw': r_raw_, 'mae_raw': m_raw_,
            'r_clean': r_clean_, 'mae_clean': m_clean_,
            'dr_clean_vs_raw': r_clean_ - r_raw_,
            'probe_top1_raw': a1_raw, 'probe_top1_clean': a1_cl,
            'probe_top3_raw': a3_raw, 'probe_top3_clean': a3_cl,
        })

    print('\n\nSUMMARY')
    print('=' * 100, flush=True)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 50)
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'), flush=True)


if __name__ == '__main__':
    main()
