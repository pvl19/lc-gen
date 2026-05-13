#!/bin/bash
# K-fold MLP probes using the MLP-pooled latent cache (all sequences).
# Mirrors kfold_latent_probe.sh but with --baseline MLP and a different
# --latents path so probe outputs go to <probe>__MLP folders.
#
# Requires extract_mlp_latents_all.sh to have been run first.
# All parameters hardcoded per project convention.

set -euo pipefail

LATENTS="output/baseline_comparison/mlp_pooled_latents_all.npz"
MOMENTS_CSV="final_pretrain/flux_moments.csv"
COMBINED_CSV="data/all_combined_metadata.csv"
OUT_ROOT="output/latent_probes"

N_FOLDS=5
HIDDEN_DIMS="256 128 64"
DROPOUT=0.1
LR=1e-3
WEIGHT_DECAY=1e-4
N_EPOCHS=80
BATCH_SIZE=512
PATIENCE=15
SEED=0

run_probe() {
    local probe="$1"
    local tag="${probe}__MLP"
    local out="${OUT_ROOT}/${tag}"

    echo
    echo "================================================================"
    echo "  probe=${probe}  baseline=MLP  -> ${out}"
    echo "================================================================"

    python scripts/kfold_latent_probe.py \
        --probe       "$probe" \
        --baseline    "MLP" \
        --latents     "$LATENTS" \
        --moments_csv "$MOMENTS_CSV" \
        --combined_csv "$COMBINED_CSV" \
        --output_dir  "$out" \
        --n_folds     "$N_FOLDS" \
        --hidden_dims  $HIDDEN_DIMS \
        --dropout     "$DROPOUT" \
        --lr          "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --n_epochs    "$N_EPOCHS" \
        --batch_size  "$BATCH_SIZE" \
        --patience    "$PATIENCE" \
        --seed        "$SEED"
}

for probe in flux_skew flux_kurt lit_prot tars_prot num_flares total_ed; do
    run_probe "$probe"
done

echo
echo "All MLP probes complete. Outputs under ${OUT_ROOT}/."

python scripts/aggregate_probe_results.py \
    --root    "$OUT_ROOT" \
    --out_md  "${OUT_ROOT}/summary.md" \
    --out_csv "${OUT_ROOT}/summary.csv"
