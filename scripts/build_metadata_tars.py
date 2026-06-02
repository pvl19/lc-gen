"""Build a TARS-Prot subset of final_pretrain/metadata.csv.

Joins the pretrain metadata against data/tars_crossmatch.csv on Gaia DR3 ID:
  - `Prot` column is overwritten with TARS `adopted_period` for stars present
    in TARS; set to NaN for non-TARS stars.
  - Adds a `tars_subset` column (`'tars'` for TARS stars, empty string
    otherwise) so kfold_age_inference.py can filter via
    `--subset_col tars_subset --subset_val tars --subset_csv ...`.

All other columns are preserved unchanged. Multi-ref rows for a star are kept
(matches the original metadata.csv shape); each ref row for a TARS star carries
the same adopted_period.

Output: final_pretrain/metadata_tars.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd


META_PATH = Path('final_pretrain/metadata.csv')
TARS_PATH = Path('data/tars_crossmatch.csv')
OUT_PATH  = Path('final_pretrain/metadata_tars.csv')


def main() -> None:
    meta = pd.read_csv(META_PATH)
    tars = pd.read_csv(TARS_PATH)

    tars_per_star = (
        tars.dropna(subset=['dr3_source_id', 'adopted_period'])
            .groupby('dr3_source_id', as_index=False)['adopted_period']
            .first()
    )
    tars_lookup = dict(zip(tars_per_star['dr3_source_id'].astype(str),
                           tars_per_star['adopted_period'].astype(float)))

    out = meta.copy()
    gaia_keys = out['GaiaDR3_ID'].astype(str)
    tars_prot = gaia_keys.map(tars_lookup)
    tars_match = tars_prot.notna()

    out['Prot'] = tars_prot
    out['tars_subset'] = np.where(tars_match.values, 'tars', '')

    n_rows = len(out)
    n_match = int(tars_match.sum())
    n_stars = out.loc[tars_match.values, 'GaiaDR3_ID'].nunique()
    n_with_age = int((tars_match & out['age_Myr'].notna()).sum())
    prot_min = float(out['Prot'].min())
    prot_max = float(out['Prot'].max())

    print(f'Input rows: {n_rows}')
    print(f'TARS-matched rows: {n_match}')
    print(f'Unique TARS stars: {n_stars}')
    print(f'TARS rows with valid age_Myr: {n_with_age}')
    print(f'Prot range (TARS): {prot_min:.3f} – {prot_max:.3f} days')

    out.to_csv(OUT_PATH, index=False)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
