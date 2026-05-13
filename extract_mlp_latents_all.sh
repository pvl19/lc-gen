#!/bin/bash
# Extract pooled MLP-baseline features over ALL sequences (no age-CSV filter).
# Produces a cache compatible with scripts/kfold_latent_probe.py so the MLP
# encoder can be probed for the same per-sector summary statistics as the RNN.
#
# Companion to extract_mlp_latents.sh (which restricts to age-labeled stars).
# All parameters hardcoded per project convention.

set -e

MLP_PATH="output/baseline_comparison/mlp_gaussian_best.pt"
H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5"
OUT_PATH="output/baseline_comparison/mlp_pooled_latents_all.npz"

K_GRID="1 8 64 720"
N_J_PER_SECTOR=1024
DEVICE="cpu"
SEED=0
LOG_EVERY=500

python scripts/extract_mlp_latents.py \
  --mlp-path "${MLP_PATH}" \
  --h5-paths ${H5_PATHS} \
  --out-path "${OUT_PATH}" \
  --k-grid ${K_GRID} \
  --n-j-per-sector ${N_J_PER_SECTOR} \
  --device "${DEVICE}" \
  --seed ${SEED} \
  --log-every ${LOG_EVERY}
