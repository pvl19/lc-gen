python scripts/plot_recon.py \
  --model_path output/simple_rnn/simple_min_gru_bi.pt \
  --output_name reconstructions_bi.png \
  --direction bi \
  --num_examples 3 \
  --seq_length 1048 \
  --hidden_size 64