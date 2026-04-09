#!/bin/bash
# Replot UMAP from saved data
#
# Edit the variables below, then run: ./replot_umap.sh

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE="output/simple_rnn/umap_plots/umap_age_model_multiscale_bprp-4.0_data.npz"
OUTPUT="output/simple_rnn/umap_plots/umap_replot_model_bprp0_1_2.png"

# For labeling purposes (these don't filter, just add to plot title/text)
N_SAMPLES="500"            # Number of samples (leave empty if unknown)
BPRP0_MIN="1.0"            # Min BPRP0 used (leave empty if no cut)
BPRP0_MAX="2.0"            # Max BPRP0 used (leave empty if no cut)

# =============================================================================
# RUN
# =============================================================================
python scripts/replot_umap.py \
    --data_file "${DATA_FILE}" \
    --output "${OUTPUT}" \
    ${N_SAMPLES:+--n_samples ${N_SAMPLES}} \
    ${BPRP0_MIN:+--bprp0_min ${BPRP0_MIN}} \
    ${BPRP0_MAX:+--bprp0_max ${BPRP0_MAX}}
