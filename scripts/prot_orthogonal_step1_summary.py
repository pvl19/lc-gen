"""Step 1 of the Prot-orthogonal-age-info investigation.

Two deliverables, both written to --output_dir:

 1. C1 / C2 / C3 apples-to-apples LOSO comparison table on ChronoFlow.
    C1 = latents (PCA4) only            (existing run on disk)
    C2 = Prot (gyro NLE) only            (LOSO-folds inherited from C1)
    C3 = latents (PCA4) + Prot           (this round's new run)
    All three restricted to the common gaia_id set so r / MAE are computed
    on identical stars.

 2. Per-PC partial correlation table + bar plot:
        partial_r(PC_k, log10_age | log10(Prot), BPRP0)   for k in 1..16
    Tells us per-PC how much age signal each direction carries that
    isn't explained by Prot + colour. The basis is the global PCA used by
    the age model (`final_model/sendit/e50/global_pca_d16.npz`).

Plain-English context: this is the first concrete measurement of "is there
age signal in the latents that Prot doesn't already explain, and where in
the PC basis does it live?" Step 2 (localize) and Step 3 (interpret) are
gated on whether Step 1 finds anything.

Usage (driven by bin/inference/prot_orthogonal_step1.sh):
    python scripts/prot_orthogonal_step1_summary.py \
        --c1_dir final_model/.../chronoflow/loso/pca4 \
        --c2_dir output/prot_orthogonal_age/.../c2_chronoflow_loso_gyro_inherited \
        --c3_dir output/prot_orthogonal_age/.../c3_chronoflow_loso_pca4_tarsProt \
        --latents final_model/sendit/e50/metaAll/latents_pretrain.npz \
        --pca_cache final_model/sendit/e50/global_pca_d16.npz \
        --metadata_csv final_pretrain/metadata.csv \
        --prot_csv final_pretrain/metadata_tars.csv \
        --subset_col ref --subset_val ChronoFlow \
        --output_dir output/prot_orthogonal_age/step1_quantify
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kfold_age_inference import load_latents_cache, load_global_pca_artifact  # noqa: E402


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _safe_mae(pred: np.ndarray, true: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(pred[ok] - true[ok])))


def _residuals_on_controls(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """OLS residuals of y on [1, controls]. Used to strip Prot+BPRP0 before
    correlating with PCs (gives partial correlation)."""
    X = np.column_stack([np.ones_like(y), controls])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def load_chronoflow_per_star(latents_npz: str, metadata_csv: str,
                              prot_csv: str, subset_col: str, subset_val: list):
    """Replicate the C1/C3 input pipeline at the per-star level: ChronoFlow
    subset, latent_max aggregation, Prot/BPRP0/age joined per gaia_id."""
    print(f'Loading latents from {latents_npz} ...')
    cache = load_latents_cache(latents_npz)
    Z = cache['latent_vectors']               # (n_rows, D)
    gids = np.asarray([str(g) for g in cache['gaia_ids']])

    meta = pd.read_csv(metadata_csv)
    meta['GaiaDR3_ID'] = meta['GaiaDR3_ID'].astype(str)
    if subset_col not in meta.columns:
        raise SystemExit(f'metadata.csv missing {subset_col!r}')
    allowed = set(meta.loc[meta[subset_col].isin(subset_val), 'GaiaDR3_ID'])
    keep_sub = np.array([g in allowed for g in gids])
    Z = Z[keep_sub]
    gids = gids[keep_sub]
    print(f'  Subset {subset_col} in {subset_val}: {Z.shape[0]} rows '
          f'({len(np.unique(gids))} unique stars)')

    # Per-star metadata join. metadata_csv may have multiple ref rows per star
    # (one per literature source); take the first non-null age and the first
    # BPRP0. Prot from prot_csv (which is metadata_tars.csv — the Prot column
    # there is the TARS-pipeline adopted_period).
    age_meta = meta.dropna(subset=['age_Myr']).drop_duplicates('GaiaDR3_ID', keep='first')
    gid_to_logage = dict(zip(age_meta['GaiaDR3_ID'],
                              np.log10(age_meta['age_Myr'].astype(float))))
    bprp_meta = meta.dropna(subset=['BPRP0']).drop_duplicates('GaiaDR3_ID', keep='first')
    gid_to_bprp = dict(zip(bprp_meta['GaiaDR3_ID'], bprp_meta['BPRP0'].astype(float)))

    prot_df = pd.read_csv(prot_csv, usecols=['GaiaDR3_ID', 'Prot'])
    prot_df['GaiaDR3_ID'] = prot_df['GaiaDR3_ID'].astype(str)
    prot_df = prot_df.dropna(subset=['Prot']).drop_duplicates('GaiaDR3_ID', keep='first')
    gid_to_prot = dict(zip(prot_df['GaiaDR3_ID'], prot_df['Prot'].astype(float)))

    # latent_max per star.
    unique_gids = np.unique(gids)
    star_lat = np.stack([Z[gids == g].max(axis=0) for g in unique_gids])
    star_logage = np.array([gid_to_logage.get(g, np.nan) for g in unique_gids])
    star_bprp   = np.array([gid_to_bprp.get(g, np.nan) for g in unique_gids])
    star_prot   = np.array([gid_to_prot.get(g, np.nan) for g in unique_gids])
    star_logprot = np.where((star_prot > 0) & np.isfinite(star_prot),
                             np.log10(star_prot), np.nan)
    print(f'  Per-star: {len(unique_gids)} stars, '
          f'{np.isfinite(star_logprot).sum()} with valid Prot.')
    return dict(gaia_ids=unique_gids, latents=star_lat, log_age=star_logage,
                bprp0=star_bprp, log_prot=star_logprot)


def per_pc_partial_correlation(stars: dict, pca, X_mean: np.ndarray,
                                X_std: np.ndarray, pca_dim: int) -> pd.DataFrame:
    ok = (np.isfinite(stars['log_age']) & np.isfinite(stars['log_prot']) &
          np.isfinite(stars['bprp0']))
    Z = stars['latents'][ok]
    logA = stars['log_age'][ok]
    logP = stars['log_prot'][ok]
    bp   = stars['bprp0'][ok]
    print(f'\nPartial-r on {ok.sum()} stars (intersection of age, Prot, BPRP0).')

    Xn = (Z - X_mean) / np.where(X_std > 0, X_std, 1.0)
    pcs = pca.transform(Xn)[:, :pca_dim]

    controls = np.column_stack([logP, bp])
    res_age = _residuals_on_controls(logA, controls)
    rows = []
    for k in range(pca_dim):
        res_pc = _residuals_on_controls(pcs[:, k], controls)
        partial_r = _safe_corr(res_pc, res_age)
        raw_r    = _safe_corr(pcs[:, k], logA)
        r_pc_p   = _safe_corr(pcs[:, k], logP)
        rows.append({
            'pc': k + 1,
            'r_raw_pc_vs_logage': raw_r,
            'partial_r_given_prot_bprp': partial_r,
            'r_pc_vs_logprot': r_pc_p,
            'var_explained': float(pca.explained_variance_ratio_[k]),
        })
    df = pd.DataFrame(rows)
    return df


def load_predictions_csv(d: Path) -> pd.DataFrame:
    """Reads kfold_predictions.csv and normalizes the columns we care about
    to (gaia_id, true_log_age, pred_log_age). Both predicted and true ages
    in these CSVs are already in log10(Myr) (column `true_log_age` and
    `pred_median`)."""
    p = d / 'kfold_predictions.csv'
    if not p.exists():
        raise SystemExit(f'missing {p}')
    df = pd.read_csv(p)
    # Different runs label the star ID column differently; normalize.
    if 'gaia_id' in df.columns:
        df['gaia_id'] = df['gaia_id'].astype(str)
    elif 'TIC_ID' in df.columns:
        # latent runs sometimes mislabel gaia_id as TIC_ID; detect by magnitude
        if df['TIC_ID'].max() > 1e10:
            df['gaia_id'] = df['TIC_ID'].astype(str)
        else:
            raise SystemExit(f'{p} has TIC_ID but values look like real TICs '
                             '(can\'t recover gaia_id without a metadata join)')
    else:
        raise SystemExit(f'{p}: no gaia_id / TIC_ID column')
    # The kfold_age_inference saver was renamed at some point — handle both
    # the old (`true_log_age`/`pred_median`) and new (`log10_true_age`/
    # `log10_pred_median`) column conventions transparently.
    true_col = 'log10_true_age' if 'log10_true_age' in df.columns else 'true_log_age'
    pred_col = 'log10_pred_median' if 'log10_pred_median' in df.columns else 'pred_median'
    if true_col not in df.columns or pred_col not in df.columns:
        raise SystemExit(f'{p}: missing {true_col} or {pred_col}')
    out = df[['gaia_id', true_col, pred_col]].copy()
    out.columns = ['gaia_id', 'true_log_age', 'pred_median']  # normalize downstream
    # Some runs (e.g. gyro_loso_inherit on metadata_tars.csv) write multiple
    # rows per star — one per literature `ref`. Collapse to one row per gaia_id
    # by averaging the predictions; without this the inner-merge below cross-
    # multiplies, inflating N and creating spurious true_log_age mismatches.
    if out['gaia_id'].duplicated().any():
        n_dup_rows = int(out['gaia_id'].duplicated().sum())
        out = out.groupby('gaia_id', as_index=False).agg(
            true_log_age=('true_log_age', 'first'),
            pred_median =('pred_median',  'mean'),
        )
        print(f'  collapsed {n_dup_rows} duplicate per-ref rows → {len(out)} unique stars')
    return out


def assemble_apples_to_apples(c1_dir, c2_dir, c3_dir) -> pd.DataFrame:
    c1 = load_predictions_csv(Path(c1_dir)).rename(
        columns={'pred_median': 'log_c1', 'true_log_age': 'log_true_c1'})
    c2 = load_predictions_csv(Path(c2_dir)).rename(
        columns={'pred_median': 'log_c2', 'true_log_age': 'log_true_c2'})
    c3 = load_predictions_csv(Path(c3_dir)).rename(
        columns={'pred_median': 'log_c3', 'true_log_age': 'log_true_c3'})
    print(f'\nLoaded predictions: C1={len(c1)}, C2={len(c2)}, C3={len(c3)}')

    df = c1.merge(c2, on='gaia_id', how='inner').merge(c3, on='gaia_id', how='inner')
    print(f'Common gaia_ids across C1/C2/C3: {len(df)}')

    # Sanity check: true_log_age should match across runs (same stars, same truth).
    for col in ('log_true_c1', 'log_true_c2', 'log_true_c3'):
        df[col] = df[col].astype(float)
    mism = ~np.isclose(df['log_true_c1'], df['log_true_c2'], rtol=1e-4) | \
           ~np.isclose(df['log_true_c1'], df['log_true_c3'], rtol=1e-4)
    if mism.any():
        print(f'  WARN: {int(mism.sum())} rows have inconsistent true_log_age across runs')
    df['log_true'] = df['log_true_c1']
    df['log_c1']   = df['log_c1'].astype(float)
    df['log_c2']   = df['log_c2'].astype(float)
    df['log_c3']   = df['log_c3'].astype(float)
    return df


def summarize_run(df: pd.DataFrame, label: str, pred_col: str) -> dict:
    return {
        'label': label,
        'n_stars': int(np.isfinite(df[pred_col] + df['log_true']).sum()),
        'pearson_r_log': _safe_corr(df[pred_col].values, df['log_true'].values),
        'mae_dex': _safe_mae(df[pred_col].values, df['log_true'].values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c1_dir', required=True,
                    help='Directory with C1 (latent-only LOSO) kfold_predictions.csv')
    ap.add_argument('--c2_dir', required=True,
                    help='Directory with C2 (gyro LOSO-inherited) kfold_predictions.csv')
    ap.add_argument('--c3_dir', required=True,
                    help='Directory with C3 (latent + Prot LOSO) kfold_predictions.csv')
    ap.add_argument('--latents', required=True,
                    help='Pretrain latents .npz cache (used for the PC partial-r table)')
    ap.add_argument('--pca_cache', required=True,
                    help='Global PCA artifact (e.g. final_model/sendit/e50/global_pca_d16.npz)')
    ap.add_argument('--pca_dim', type=int, default=16,
                    help='Number of PCs to score for partial correlation (default: 16)')
    ap.add_argument('--metadata_csv', required=True)
    ap.add_argument('--prot_csv', required=True,
                    help='CSV with GaiaDR3_ID + Prot (e.g. final_pretrain/metadata_tars.csv)')
    ap.add_argument('--subset_col', default='ref')
    ap.add_argument('--subset_val', nargs='+', default=['ChronoFlow'])
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # ── (1) apples-to-apples C1/C2/C3 table ────────────────────────────────
    print('\n=== Assembling apples-to-apples C1/C2/C3 table ===')
    merged = assemble_apples_to_apples(args.c1_dir, args.c2_dir, args.c3_dir)
    merged.to_csv(out / 'apples_to_apples_predictions.csv', index=False)

    rows = [
        summarize_run(merged, 'C1: latents (PCA4) only',          'log_c1'),
        summarize_run(merged, 'C2: Prot (gyro NLE) only',          'log_c2'),
        summarize_run(merged, 'C3: latents (PCA4) + Prot',         'log_c3'),
    ]
    summary = pd.DataFrame(rows)
    summary['delta_r_vs_C1']   = summary['pearson_r_log'] - rows[0]['pearson_r_log']
    summary['delta_mae_vs_C1'] = summary['mae_dex']       - rows[0]['mae_dex']
    summary.to_csv(out / 'c1_c2_c3_table.csv', index=False)
    print('\n=== Apples-to-apples LOSO (common stars only) ===')
    print(summary.to_string(index=False))

    # ── (2) per-PC partial correlation ─────────────────────────────────────
    print('\n=== Per-PC partial correlation: partial_r(PC_k, log10_age | log10(Prot), BPRP0) ===')
    stars = load_chronoflow_per_star(args.latents, args.metadata_csv,
                                      args.prot_csv, args.subset_col,
                                      args.subset_val)
    pca, X_mean, X_std = load_global_pca_artifact(args.pca_cache, args.pca_dim)
    partial = per_pc_partial_correlation(stars, pca, X_mean, X_std, args.pca_dim)
    pc_dir = out / 'partial_corr_per_pc'
    pc_dir.mkdir(parents=True, exist_ok=True)
    partial.to_csv(pc_dir / 'per_pc_partial_correlation.csv', index=False)
    print('\n' + partial.to_string(index=False))

    # Bar plot — raw r and partial r side-by-side, plus PC↔log(Prot) marker
    fig, ax = plt.subplots(figsize=(0.55 * args.pca_dim + 2, 4.5))
    x = np.arange(args.pca_dim)
    w = 0.4
    ax.bar(x - w/2, partial['r_raw_pc_vs_logage'],         w,
           label=r'$r(\mathrm{PC}_k,\, \log_{10}\mathrm{age})$', color='#888888')
    ax.bar(x + w/2, partial['partial_r_given_prot_bprp'],  w,
           label=r'$r(\mathrm{PC}_k,\, \log_{10}\mathrm{age}\,|\,\log_{10}\mathrm{Prot},\,\mathrm{BPRP_0})$',
           color='#cc4444')
    ax.scatter(x, partial['r_pc_vs_logprot'], marker='x', color='#1f77b4',
               label=r'$r(\mathrm{PC}_k,\, \log_{10}\mathrm{Prot})$')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f'PC{i+1}' for i in range(args.pca_dim)],
                                          rotation=0, fontsize=8)
    ax.set_ylabel('Pearson r')
    ax.set_title('Per-PC age signal: raw vs Prot-controlled (ChronoFlow per-star)')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(pc_dir / 'per_pc_partial_correlation.png', dpi=150)
    plt.close(fig)
    print(f'\nSaved: {pc_dir / "per_pc_partial_correlation.png"}')

    # Stash a tiny JSON readout with the headline numbers, for grep-ability.
    headline = {
        'population': 'ChronoFlow per-star (LOSO)',
        'common_n_stars': int(len(merged)),
        'c1_r': rows[0]['pearson_r_log'], 'c1_mae': rows[0]['mae_dex'],
        'c2_r': rows[1]['pearson_r_log'], 'c2_mae': rows[1]['mae_dex'],
        'c3_r': rows[2]['pearson_r_log'], 'c3_mae': rows[2]['mae_dex'],
        'delta_r_c3_vs_c1':   rows[2]['pearson_r_log'] - rows[0]['pearson_r_log'],
        'delta_mae_c3_vs_c1': rows[2]['mae_dex']       - rows[0]['mae_dex'],
        'best_partial_pc':    int(partial.iloc[partial['partial_r_given_prot_bprp']
                                                .abs().idxmax()]['pc']),
        'best_partial_r':     float(partial['partial_r_given_prot_bprp']
                                      .abs().max()),
    }
    with open(out / 'headline.json', 'w') as f:
        json.dump(headline, f, indent=2)
    print('\n=== Headline ===')
    print(json.dumps(headline, indent=2))


if __name__ == '__main__':
    main()
