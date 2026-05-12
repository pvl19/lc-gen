#!/bin/bash
# Baseline comparison: RNN+flow vs. local-window MLP baseline.
# All parameters hardcoded below — edit this file to change a run.
# Outputs land in output/baseline_comparison/.

set -e

H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5"
OUT_DIR="output/baseline_comparison"

# --- Split (gaia_id partitioning) ---
EVAL_MOD=10            # 10% of stars in eval set
SANITY_MOD=100         # 1% of stars in sanity-val set
SANITY_RESIDUE=7

# --- MLP training ---
DEVICE="cpu"           # "cpu" or "cuda"
EPOCHS=5
LR=1e-3
K_MAX=720              # log-uniform offset upper bound (matches RNN training)
C=32                   # context window size per side
CONTEXT_DIM=136        # MLP output context dim (matches RNN output_size = 2*64+8)
HIDDEN_DIMS="128 128"
SEED=0
LOG_EVERY=200
GRAD_ACCUM=1
N_TARGETS_PER_SEQ=1024   # random j-positions sampled per sequence per step (0 = all)

# --- Smoke-test caps. Set to 0 for full runs. ---
MAX_TRAIN_SEQS=0
MAX_SANITY_SEQS=0

# --- Eval ---
RNN_PATH="final_model/parallel_fixed/e110/best_model.pt"
N_EVAL_TARGETS_PER_SEQ=256   # j positions subsampled per (sequence, k)
N_FLOW_SAMPLES=0             # 0 = NLL only; bump (e.g. 128) for MAE-on-median + coverage
MAX_EVAL_SEQS=0              # 0 = full eval-10%, else smoke cap
EVAL_LOG_EVERY=100

mkdir -p "${OUT_DIR}"

# 1. Build deterministic gaia_id split (idempotent — rerunning is fine).
python scripts/baseline_comparison.py split \
  --h5-paths ${H5_PATHS} \
  --eval-mod ${EVAL_MOD} \
  --sanity-mod ${SANITY_MOD} \
  --sanity-residue ${SANITY_RESIDUE} \
  --out-dir "${OUT_DIR}"

# 2. Train Gaussian MLP baseline.
python scripts/baseline_comparison.py train_gaussian \
  --split-path "${OUT_DIR}/split.json" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --K-max ${K_MAX} \
  --C ${C} \
  --context-dim ${CONTEXT_DIM} \
  --hidden-dims ${HIDDEN_DIMS} \
  --seed ${SEED} \
  --log-every ${LOG_EVERY} \
  --grad-accum ${GRAD_ACCUM} \
  --max-train-seqs ${MAX_TRAIN_SEQS} \
  --max-sanity-seqs ${MAX_SANITY_SEQS} \
  --n-targets-per-seq ${N_TARGETS_PER_SEQ}

# 3. Evaluate all methods on the eval-10% split.
python scripts/baseline_comparison.py eval \
  --split-path "${OUT_DIR}/split.json" \
  --out-dir "${OUT_DIR}" \
  --mlp-path "${OUT_DIR}/mlp_gaussian_best.pt" \
  --rnn-path "${RNN_PATH}" \
  --device "${DEVICE}" \
  --seed ${SEED} \
  --K-max ${K_MAX} \
  --n-eval-targets-per-seq ${N_EVAL_TARGETS_PER_SEQ} \
  --n-flow-samples ${N_FLOW_SAMPLES} \
  --max-eval-seqs ${MAX_EVAL_SEQS} \
  --log-every ${EVAL_LOG_EVERY}
