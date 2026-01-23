# Set USE_METADATA=false to plot without stellar parameters
USE_METADATA=${USE_METADATA:-true}

python scripts/plot_recon.py \
  --model_path output/simple_rnn/models/baseline_v18_stellar_params.pt \
  --output_name reconstructions_v18_stellar_params.png \
  --direction bi \
  --num_examples 3 \
  --seq_length 1024 \
  --hidden_size 64 \
  --min_size 5 \
  --max_size 100 \
  --mask_portion 0.4 \
  --random_seed 69 \
  --mode parallel \
  --lags 1 16 64 \
  --use_flow \
  ${USE_METADATA:+--use_metadata} \
  --flow_mode mean  # Options: mode, mean, sample