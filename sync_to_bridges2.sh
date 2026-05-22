#!/usr/bin/env bash
# Sync required files to PSC Bridges-2 via scp.
# Usage: ./sync_to_bridges2.sh [bridges2-username]
#
# rsync is not available on the Bridges-2 login nodes. Transfers go through
# PSC's dedicated data-transfer host, data.bridges2.psc.edu — a restricted
# shell that allows only file-transfer commands, so scp works there while an
# interactive 'rsync'/'which' does not. scp needs nothing on the remote but
# the SSH server.
#
# scp has no delta-skip (unlike rsync --ignore-existing): every listed file is
# re-uploaded in full each run. The source files are tiny; the data H5s are
# large, so comment out the data block once they are uploaded and unchanged.
#
# TIP: run `ssh-copy-id ${BRIDGES2_USER}@data.bridges2.psc.edu` once to set up
# key auth — otherwise scp prompts for your password on every call below.

BRIDGES2_USER="${1:-pvanlane}"
XFER_HOST="${BRIDGES2_USER}@data.bridges2.psc.edu"   # file transfers (scp)
LOGIN_HOST="bridges2"                                 # interactive / sbatch (your ssh alias)
REMOTE_DIR="lcgen"

echo "========================================"
echo "Syncing to ${XFER_HOST}:${REMOTE_DIR}"
echo "========================================"

# --- Source code ---
echo "Syncing source code..."
scp src/lcgen/train_simple_rnn.py \
    "${XFER_HOST}:${REMOTE_DIR}/src/lcgen/"

scp src/lcgen/models/simple_min_gru.py \
    src/lcgen/models/TimeSeriesDataset.py \
    src/lcgen/models/MetadataAgePredictor.py \
    src/lcgen/models/conv_models.py \
    src/lcgen/models/lightweight_conv.py \
    "${XFER_HOST}:${REMOTE_DIR}/src/lcgen/models/"

scp src/lcgen/utils/loss.py \
    src/lcgen/utils/mask.py \
    src/lcgen/utils/metadata_masking.py \
    src/lcgen/utils/run_log.py \
    src/lcgen/utils/trunc_data.py \
    "${XFER_HOST}:${REMOTE_DIR}/src/lcgen/utils/"

# --- SLURM script ---
echo "Syncing SLURM script..."
scp slurm_bridges2.sh "${XFER_HOST}:${REMOTE_DIR}/"

# --- Spectra sidecar files (conv-encoder branch only) — NOT synced ---
# scp final_pretrain/timeseries_pretrain_spectra.h5 \
#     final_pretrain/timeseries_exop_hosts_spectra.h5 \
#     "${XFER_HOST}:${REMOTE_DIR}/final_pretrain/"

# --- Data (large — all three H5 files re-uploaded in full) ---
echo "Syncing data files (large — this will take a while)..."
scp final_pretrain/timeseries_pretrain.h5 \
    final_pretrain/timeseries_exop_hosts.h5 \
    final_pretrain/timeseries_thickdisk.h5 \
    "${XFER_HOST}:${REMOTE_DIR}/final_pretrain/"

echo "========================================"
echo "Done. To submit the job (on the LOGIN node, not the transfer node):"
echo "  ssh ${LOGIN_HOST}"
echo "  cd \$HOME/lcgen"
echo "  mkdir -p slurm_logs checkpoints/resume"
echo "  sbatch slurm_bridges2.sh"
echo "========================================"
