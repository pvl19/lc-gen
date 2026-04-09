# Resume Training from Checkpoint

The training script now supports resuming from a previous checkpoint. This allows you to:
- Train longer (e.g., from 20 epochs to 100 epochs)
- Continue training after early stopping
- Resume interrupted jobs

## How It Works

When resuming:
1. **Model state** is loaded from the checkpoint `.pt` file
2. **Optimizer state** is restored (learning rate, momentum, etc.)
3. **Previous losses** are loaded from `losses_per_epoch.npz`
4. **Epoch counter** reflects total epochs (previous + new)
5. **Early stopping** uses best validation loss from previous training

## Usage

### Option 1: Using slurm.sh (SDSC Expanse)

1. Edit [slurm.sh](slurm.sh) and set the `RESUME_FROM` variable (around line 26):
   ```bash
   RESUME_FROM="output/46136857-lcgen-baseline-metadata-xchannels/baseline_metadata_xchannels.pt"
   ```

2. Update `--epochs` to your new target:
   ```bash
   --epochs 100  # Was 20, now training to 100 total
   ```

3. Submit the job:
   ```bash
   sbatch slurm.sh
   ```

### Option 2: Local training

```bash
python src/lcgen/train_simple_rnn.py \
    --input data/timeseries_x.h5 \
    --output_dir output/continued \
    --output_name model_continued.pt \
    --epochs 100 \
    --use_conv_channels \
    --conv_encoder_type unet \
    --use_metadata \
    --patience 10 \
    --resume_from output/previous_run/model.pt
```

## What Gets Preserved

When resuming training, these are preserved:
- ✅ Model weights (all layers, including conv encoders)
- ✅ Optimizer state (Adam momentum, learning rate schedule position)
- ✅ Training history (loss curves from epoch 1 onwards)
- ✅ Best validation loss (for early stopping)

## Important Notes

1. **Epoch count**: If you trained for 20 epochs and resume with `--epochs 50`, you'll train for 30 MORE epochs (20 → 50 total)

2. **Early stopping**: The best validation loss is preserved, so early stopping will compare against the best loss from all epochs (previous + current)

3. **Learning rate**: The OneCycleLR scheduler restarts for the remaining epochs. This is intentional - you get a fresh cycle for the continuation.

4. **Output files**: The checkpoint and losses_per_epoch.npz are updated with combined results from both sessions

5. **Same architecture required**: The checkpoint must match the model architecture (same hidden_size, use_metadata, use_conv_channels, etc.)

## Example: Training from 20 to 100 Epochs

Your model was trained for 20 epochs and stopped early. To continue to 100 epochs:

1. Find your checkpoint (e.g., `output/46136857-lcgen-baseline-metadata-xchannels/baseline_metadata_xchannels.pt`)

2. Edit [slurm.sh](slurm.sh):
   ```bash
   # Line 26:
   RESUME_FROM="output/46136857-lcgen-baseline-metadata-xchannels/baseline_metadata_xchannels.pt"

   # Line 89 (or wherever --epochs is):
   --epochs 100 \
   ```

3. Submit:
   ```bash
   sbatch slurm.sh
   ```

The training will:
- Load the model from epoch 20
- Continue training epochs 21, 22, 23, ... up to 100
- Save the final model with all 100 epochs of training
- The losses_per_epoch.npz will have 100 entries

## Checking if Resume Worked

When the job starts, look for these messages in the SLURM output:

```
============================================================
RESUMING FROM CHECKPOINT: checkpoint/baseline_metadata_xchannels.pt
============================================================
✓ Loaded model state from checkpoint
✓ Loaded optimizer state from checkpoint
✓ Loaded previous losses: 20 epochs already completed
  Last train loss: 1.117498
  Last val loss: 1.173825

Resuming training from epoch 21 to epoch 100
============================================================
```

If you see this, the resume worked correctly!

## Troubleshooting

**Problem**: "Checkpoint not found" error

**Solution**: Make sure the path in `RESUME_FROM` is relative to `PROJECT_DIR` (e.g., `output/46136857-.../model.pt`, not an absolute path)

---

**Problem**: Model architecture mismatch error

**Solution**: Ensure the resume training uses the same flags as the original training:
- Same `--use_metadata`
- Same `--use_conv_channels`
- Same `--conv_encoder_type`
- Same `--hidden_size`
- Same `--direction` (bi, forward, backward)

---

**Problem**: Training starts from epoch 1 instead of resuming

**Solution**: Check that `losses_per_epoch.npz` exists in the same directory as the `.pt` checkpoint file
