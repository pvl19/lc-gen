#!/bin/bash
# K-fold MLP probes: test how well the autoencoder latent space recovers
# four per-light-curve summary statistics, with random/shuffled baselines.
#
# All parameters are hardcoded per project convention. Comment out any
# (probe, baseline) row to skip it.

set -euo pipefail

LATENTS="final_model/parallel_fixed/e60/latents.npz"
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
    local baseline="$2"
    local tag="$probe"
    [ "$baseline" != "none" ] && tag="${probe}__${baseline}"
    local out="${OUT_ROOT}/${tag}"

    echo
    echo "================================================================"
    echo "  probe=${probe}  baseline=${baseline}  -> ${out}"
    echo "================================================================"

    python scripts/kfold_latent_probe.py \
        --probe       "$probe" \
        --baseline    "$baseline" \
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

for probe in flux_skew flux_kurt lit_prot tars_prot; do
    for baseline in none gaussian shuffle; do
        run_probe "$probe" "$baseline"
    done
done

echo
echo "All probes complete. Outputs under ${OUT_ROOT}/."
