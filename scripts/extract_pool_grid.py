#!/usr/bin/env python3
"""Extract the multiscale POOL-CONFIG GRID from a single checkpoint in one pass.

The RNN forward (the expensive part) runs once per light curve; each pool config
is just a cheap reduction of the same hidden states (see extract_latent_vectors'
`pool_configs` path). Emits one latent npz per config per H5 category, laid out
as <OUT_BASE>/<config>/latents_<category>.npz so sector_partial_probe.py can read
each config directory directly.

Grid (diff fixed at the physically-mandatory Δt-weighted 'dt'):
    glob_mode ∈ {uniform, voronoi, voronoi_capped}
    seg_mode  ∈ {equal_count, equal_time_carry}
  + one 'legacy-timeaware' config (voronoi / equal_time zero-fill / unweighted)
    extracted in the SAME run as an apples-to-apples reference for the broken
    time-aware pool that produced the e110_timeaware cache.

All parameters hardcoded (project convention). Point MODEL_PATH/OUT_BASE/H5 at a
different checkpoint to grid another model.
"""
import sys
from pathlib import Path

import torch

ROOT = Path('/Users/philvanlane/Documents/lc_ae')
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from plot_umap_latent import (  # noqa: E402
    load_model, extract_latent_vectors, load_ages, save_latents_cache, METADATA_FEATURES)

# --- Target checkpoint + outputs ---
MODEL_PATH = ROOT / 'final_model/parallel_fixed/e110/best_model.pt'
OUT_BASE   = ROOT / 'final_model/parallel_fixed/e110_poolgrid'
# e110 predates the thick-disk set and its cache is pretrain+hosts only — match it.
H5 = [
    ('pretrain', ROOT / 'final_pretrain/timeseries_pretrain.h5'),
    ('hosts',    ROOT / 'final_pretrain/timeseries_exop_hosts.h5'),
]
AGE_CSV = [str(ROOT / 'final_pretrain/metadata.csv'),
           str(ROOT / 'final_pretrain/host_all_metadata.csv')]

# Extraction params — canonical plot_umap.sh settings (trim has negligible effect
# here; verified trim=0 vs 10 changes the latent by <0.1%).
TRIM_EDGES = 10
MINMAX_EDGE_SKIP = 100
BATCH = 16

GRID = []
for glob in ('uniform', 'voronoi', 'voronoi_capped'):
    for seg in ('equal_count', 'equal_time_carry'):
        GRID.append({'name': f'glob-{glob}__seg-{seg}',
                     'glob_mode': glob, 'seg_mode': seg, 'diff_mode': 'dt'})
# In-run reference: the broken time-aware pool (matches the e110_timeaware cache's
# intent), so corrected configs are compared on an identical pipeline.
GRID.append({'name': 'legacy-timeaware',
             'glob_mode': 'voronoi', 'seg_mode': 'equal_time', 'diff_mode': 'unweighted'})


def main():
    dev = torch.device('cpu')
    model = load_model(str(MODEL_PATH), device=dev, hidden_size=64, direction='bi',
                       mode='parallel', use_flow=True,
                       num_meta_features=len(METADATA_FEATURES))
    print(f'Pool grid ({len(GRID)} configs): {[c["name"] for c in GRID]}')

    for cat, h5 in H5:
        print(f'\n=== {cat}: {h5} ===')
        out = extract_latent_vectors(
            model, str(h5), dev, batch_size=BATCH, pooling_mode='multiscale',
            use_metadata=True, trim_edges=TRIM_EDGES, minmax_edge_skip=MINMAX_EDGE_SKIP,
            pool_configs=GRID)
        (ages, bprp0, bprp0_err, mg, mg_err, mem_prob,
         gaia_ids, tic_ids, sectors) = load_ages(str(h5), AGE_CSV, sample_indices=None)
        for cfg in GRID:
            path = OUT_BASE / cfg['name'] / f'latents_{cat}.npz'
            save_latents_cache(str(path), out[cfg['name']], ages, bprp0,
                               gaia_ids, tic_ids, sectors,
                               bprp0_err=bprp0_err, mg=mg, mg_err=mg_err, mem_prob=mem_prob)
            print(f'  saved {path}  {out[cfg["name"]].shape}')
        del out
    print('\nDONE — probe with:')
    print('  python scripts/sector_partial_probe.py \\')
    print('    ' + ' '.join(f'{OUT_BASE}/{c["name"]}' for c in GRID) + ' \\')
    print('    e110')


if __name__ == '__main__':
    main()
