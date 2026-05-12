#!/bin/bash
# Baseline comparison — EVAL.
# Scores mlp_gaussian, rnn_flow, nn_mean, window_mean on the eval-10% split.
# Requires baseline_comparison.sh to have been run first (produces split.json
# and mlp_gaussian_best.pt). All parameters hardcoded below.
#
# Output: output/baseline_comparison/summary.csv

set -e

OUT_DIR="output/baseline_comparison"

DEVICE="cpu"
SEED=0
K_MAX=720
C=32                            # must match the trained MLP's C

# --- Naive-baseline sigma calibration ---
FIT_STARS=500                   # training stars used to fit sigma_k for nn_mean / window_mean
FIT_N_TARGETS=256

MLP_PATH="${OUT_DIR}/mlp_gaussian_best.pt"
RNN_PATH="final_model/parallel_fixed/e110/best_model.pt"

N_EVAL_TARGETS_PER_SEQ=256      # j positions subsampled per (sequence, k)
N_FLOW_SAMPLES=0                # 0 = NLL only (cheap). Set to 128 for MAE-on-median + coverage on rnn_flow.
MAX_EVAL_SEQS=0                 # 0 = full eval-10%, else smoke cap
LOG_EVERY=100

# 1. Calibrate per-k Gaussian sigma for nn_mean / window_mean on training residuals.
python scripts/baseline_comparison.py fit_baselines \
  --split-path "${OUT_DIR}/split.json" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --seed ${SEED} \
  --K-max ${K_MAX} \
  --C ${C} \
  --fit-stars ${FIT_STARS} \
  --n-targets-per-seq ${FIT_N_TARGETS}

# 2. Score all methods on the eval-10% split.
python scripts/baseline_comparison.py eval \
  --split-path "${OUT_DIR}/split.json" \
  --out-dir "${OUT_DIR}" \
  --mlp-path "${MLP_PATH}" \
  --rnn-path "${RNN_PATH}" \
  --device "${DEVICE}" \
  --seed ${SEED} \
  --K-max ${K_MAX} \
  --n-eval-targets-per-seq ${N_EVAL_TARGETS_PER_SEQ} \
  --n-flow-samples ${N_FLOW_SAMPLES} \
  --max-eval-seqs ${MAX_EVAL_SEQS} \
  --log-every ${LOG_EVERY}

# 3. Render NLL / MAE / RMSE vs. k plots from summary.csv.
python scripts/baseline_comparison.py plot \
  --summary-csv "${OUT_DIR}/summary.csv" \
  --out-dir "${OUT_DIR}"
