#!/bin/bash
# Re-extract latents for sectors 97 and 98 with each ~60-day light curve split
# in half (so each becomes two ~30-day half-sector latents, in-distribution with
# the rest of the dataset).
#
# Writes one sidecar npz per H5 alongside the existing latents — does NOT
# overwrite them. Each original sector-97/98 row produces TWO output rows
# (subsector=0 first half, subsector=1 second half), sharing gaia_id / tic_id /
# sector / age / bprp0 / etc. Schema is otherwise the canonical kfold format.
#
# Defaults mirror plot_umap.sh (sendit/e100 official bank, locked-in pooling:
# uniform / equal_count / dt, trim_edges=10). The model MUST match whichever
# bank you intend to merge into — e100 split latents only make sense if the
# main e100 latents were also extracted with e100.

MODEL_PATH="final_model/sendit/e100/best_model.pt"
LATENT_BASE="final_model/sendit/e100"

H5_PATHS="final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5 final_pretrain/timeseries_thickdisk.h5"
AGE_CSVS="final_pretrain/metadata.csv final_pretrain/host_all_metadata.csv"

# One save path per H5 file, IN THE SAME ORDER as H5_PATHS above.
SAVE_SPLIT_LATENTS="${LATENT_BASE}/latents_pretrain_s97s98_split.npz \
${LATENT_BASE}/latents_hosts_s97s98_split.npz \
${LATENT_BASE}/latents_thickdisk_s97s98_split.npz"

TARGET_SECTORS="97 98"

# Must match training and the original extraction. Default 10 strips TESS
# sector-edge artifacts at the *original* sector boundaries; the new midpoint
# split is interior, so no trim is applied there.
TRIM_EDGES=10

# Locked-in pooling defaults (mirror plot_umap.sh / extract_latent_vectors).
GLOB_MODE="uniform"
SEG_MODE="equal_count"
DIFF_WEIGHT_MODE="dt"
MINMAX_EDGE_SKIP=0
BATCH_SIZE=16

mkdir -p "${LATENT_BASE}"

python scripts/extract_split_latents_s97_98.py \
  --model_path "${MODEL_PATH}" \
  --h5_path ${H5_PATHS} \
  --age_csv_path ${AGE_CSVS} \
  --save_split_latents ${SAVE_SPLIT_LATENTS} \
  --target_sectors ${TARGET_SECTORS} \
  --hidden_size 64 \
  --direction bi \
  --mode parallel \
  --use_flow \
  --use_metadata \
  --batch_size ${BATCH_SIZE} \
  --trim_edges ${TRIM_EDGES} \
  --minmax_edge_skip ${MINMAX_EDGE_SKIP} \
  --diff_weight_mode ${DIFF_WEIGHT_MODE} \
  --glob_mode ${GLOB_MODE} \
  --seg_mode ${SEG_MODE}
