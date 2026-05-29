#!/bin/bash
# Drop sector 97/98 rows from each main latents npz and replace with the
# per-half split latents. Writes new merged files alongside the main bank —
# does NOT overwrite. After verifying the merged files, you can mv them over
# the originals (or just point plot_umap.sh's LOAD_LATENTS at the *_merged.npz
# paths and leave the originals untouched).
#
# Both main and split latents MUST come from the same model checkpoint, or the
# sector 97/98 rows will sit in a different region of latent space than the
# rest of the bank for reasons unrelated to sequence length.

LATENT_BASE="final_model/sendit/e100"

MAIN="${LATENT_BASE}/latents_pretrain.npz \
${LATENT_BASE}/latents_hosts.npz \
${LATENT_BASE}/latents_thickdisk.npz"

SPLIT="${LATENT_BASE}/latents_pretrain_s97s98_split.npz \
${LATENT_BASE}/latents_hosts_s97s98_split.npz \
${LATENT_BASE}/latents_thickdisk_s97s98_split.npz"

OUT="${LATENT_BASE}/latents_pretrain_merged.npz \
${LATENT_BASE}/latents_hosts_merged.npz \
${LATENT_BASE}/latents_thickdisk_merged.npz"

python scripts/merge_split_latents_s97_98.py \
  --main ${MAIN} \
  --split ${SPLIT} \
  --out ${OUT} \
  --target_sectors 97 98
