#!/bin/bash
# Extract pooled MLP-baseline features for downstream age inference.
# Forwards mlp_gaussian_best.pt at multiple k for each labeled sequence,
# pools last-hidden features with mean+std, writes a latents npz cache
# compatible with kfold_age_inference.py --load_latents.
#
# See docs/plans/2026-05-12_mlp-pooled-age-inference.md for context.
# All parameters hardcoded here per project convention.

set -e

MLP_PATH="output/baseline_comparison/mlp_gaussian_best.pt"
H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5"
AGE_CSV="final_pretrain/all_ages.csv"
OUT_PATH="output/baseline_comparison/mlp_pooled_latents.npz"

# k grid spans the training distribution (1..720, log-uniform). Features at
# each k are averaged before pooling across j -- marginalizes out k so the
# representation is k-independent, matching the RNN hidden state.
K_GRID="1 8 64 720"

N_J_PER_SECTOR=1024     # dense subsample of valid j positions per sector
DEVICE="cpu"
SEED=0
LOG_EVERY=200

python scripts/extract_mlp_latents.py \
  --mlp-path "${MLP_PATH}" \
  --h5-paths ${H5_PATHS} \
  --age-csv "${AGE_CSV}" \
  --out-path "${OUT_PATH}" \
  --k-grid ${K_GRID} \
  --n-j-per-sector ${N_J_PER_SECTOR} \
  --device "${DEVICE}" \
  --seed ${SEED} \
  --log-every ${LOG_EVERY}
