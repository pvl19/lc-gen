#!/bin/bash
# Step 1 of the Prot-orthogonal-age-info investigation
# (docs/plans/2026-06-30_prot-orthogonal-age-info.md).
#
# Trains C3 = ChronoFlow LOSO PCA4 with log10(tars_Prot) as the 4th NSF
# context dim, mirroring the canonical C1 run at
#   final_model/sendit/e50/age_inference/chronoflow/loso/pca4/
# but with --use_prot --prot_csv final_pretrain/metadata_tars.csv.
#
# All other hyperparameters match C1 exactly so the only delta is "Prot in
# the context vs not in the context." Apples-to-apples subsetting (drop the
# stars with missing Prot from BOTH C1 and C3 prediction CSVs) is performed
# after training by scripts/prot_orthogonal_step1_summary.py.
#
# Outputs land in output/prot_orthogonal_age/step1_quantify/.

set -u  # fail loud on unset vars

OUT_ROOT=output/prot_orthogonal_age/step1_quantify
C3_DIR=${OUT_ROOT}/c3_chronoflow_loso_pca4_tarsProt
mkdir -p ${C3_DIR}

LAT=final_model/sendit/e50/metaAll/latents_pretrain.npz
HOSTS=final_model/sendit/e50/metaAll/latents_hosts.npz
THICK=final_model/sendit/e50/metaAll/latents_thickdisk.npz
PCA_CACHE=final_model/sendit/e50/global_pca_d16.npz

echo "############## C3: ChronoFlow LOSO pca4 + log10(tars_Prot) ##############"
python scripts/kfold_age_inference.py \
  --load_latents ${LAT} \
  --age_csv final_pretrain/metadata.csv --override_ages_from_csv \
  --subset_col ref --subset_val ChronoFlow \
  --subset_csv final_pretrain/metadata.csv \
  --pooling_mode multiscale --star_aggregation latent_max \
  --use_metadata --loga_grid_size 1000 --seed 42 \
  --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --encoder_type pca --training_stages joint --n_epochs 100 \
  --pca_latent_pool ${LAT} ${HOSTS} ${THICK} \
  --pca_cache ${PCA_CACHE} --pca_cache_max_dim 16 \
  --pca_dim 4 --n_folds 10 --loso \
  --use_prot --prot_csv final_pretrain/metadata_tars.csv \
  --output_dir ${C3_DIR} \
  || { echo "  !! C3 run FAILED"; exit 1; }

C2_DIR=${OUT_ROOT}/c2_chronoflow_loso_gyro_inherited
echo ""
echo "############## C2: ChronoFlow LOSO gyro (folds inherited from C1) ##############"
python scripts/kfold_gyro_loso_inherit.py \
  --age_csv final_pretrain/metadata_tars.csv \
  --metadata_csv final_pretrain/metadata.csv \
  --inherit_folds_from final_model/sendit/e50/age_inference/chronoflow/loso/pca4/kfold_predictions.csv \
  --output_dir ${C2_DIR} \
  --seed 42 --batch_size 64 --lr 1e-3 --lr_decay_rate 0.97 \
  --flow_transforms 6 --flow_hidden_dims 64 64 \
  --n_epochs 100 --loga_grid_size 1000 \
  || { echo "  !! C2 LOSO gyro inherited FAILED"; exit 1; }

echo ""
echo "############## Step 1 summary (C1/C2/C3 + per-PC partial-r) ##############"
python scripts/prot_orthogonal_step1_summary.py \
  --c3_dir ${C3_DIR} \
  --c1_dir final_model/sendit/e50/age_inference/chronoflow/loso/pca4 \
  --c2_dir ${C2_DIR} \
  --latents ${LAT} --pca_cache ${PCA_CACHE} \
  --metadata_csv final_pretrain/metadata.csv \
  --prot_csv final_pretrain/metadata_tars.csv \
  --subset_col ref --subset_val ChronoFlow \
  --output_dir ${OUT_ROOT}
