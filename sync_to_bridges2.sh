#!/usr/bin/env bash
# Sync required files to PSC Bridges-2.
# Usage: ./sync_to_bridges2.sh [bridges2-username]
#
# Set BRIDGES2_USER below or pass as first argument.
# Assumes your SSH key is configured for bridges2.psc.edu.

BRIDGES2_USER="${1:-pvanlane}"
REMOTE="bridges2"
REMOTE_DIR="lcgen"

echo "========================================"
echo "Syncing to ${REMOTE}:${REMOTE_DIR}"
echo "========================================"

# --- Source code ---
echo "Syncing source code..."
rsync -avz \
    src/lcgen/train_simple_rnn.py \
    "${REMOTE}:${REMOTE_DIR}/src/lcgen/"

rsync -avz \
    src/lcgen/models/simple_min_gru.py \
    src/lcgen/models/TimeSeriesDataset.py \
    src/lcgen/models/MetadataAgePredictor.py \
    src/lcgen/models/conv_models.py \
    src/lcgen/models/lightweight_conv.py \
    "${REMOTE}:${REMOTE_DIR}/src/lcgen/models/"

rsync -avz \
    src/lcgen/utils/loss.py \
    src/lcgen/utils/mask.py \
    src/lcgen/utils/run_log.py \
    src/lcgen/utils/trunc_data.py \
    "${REMOTE}:${REMOTE_DIR}/src/lcgen/utils/"

# --- SLURM script ---
echo "Syncing SLURM script..."
rsync -avz slurm_bridges2.sh "${REMOTE}:${REMOTE_DIR}/"

# --- Data (large — skipped if already present with --ignore-existing) ---
# echo "Syncing data files (this may take a while)..."
# rsync -avz --progress --ignore-existing \
#     final_pretrain/timeseries_pretrain.h5 \
#     final_pretrain/timeseries_exop_hosts.h5 \
#     "${REMOTE}:${REMOTE_DIR}/final_pretrain/"

echo "========================================"
echo "Done. To submit the job:"
echo "  ssh ${REMOTE}"
echo "  cd \$HOME/lcgen"
echo "  mkdir -p slurm_logs checkpoints/resume"
echo "  sbatch slurm_bridges2.sh"
echo "========================================"
