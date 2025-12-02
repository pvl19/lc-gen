python src/lcgen/train_simple_rnn.py \
  --input data/timeseries.h5 \
  --random_seed 19 \
  --max_length 1048 \
  --num_samples 100 \
  --output_name simple_min_gru_rolling.pt \
  --epochs 40 \
  --lr 1e-3