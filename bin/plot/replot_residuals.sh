#!/bin/bash
# Replot residual histograms from saved combined_predictions.csv
# Edit replot_residuals.py directly for plot styling

# ============== CONFIGURATION ==============
INPUT_CSV="output/age_predictor/comparison_v2_multiscale_vs_gyro/combined_predictions.csv"
OUTPUT_PATH="output/age_predictor/comparison_v2_multiscale_vs_gyro/residuals_replot.png"
# ============================================

python scripts/replot_residuals.py \
    --input_csv "$INPUT_CSV" \
    --output_path "$OUTPUT_PATH"
