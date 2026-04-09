# Quick Reference: Resume Training on SDSC Expanse

## Folder Structure on Expanse

```
$HOME/lcgen/
├── checkpoints/resume/          # ← Put your resume files here
│   ├── model.pt                 # ← Your checkpoint (required)
│   └── losses_per_epoch.npz     # ← Your losses (required)
└── slurm.sh                     # ← Edit this to set RESUME_FROM
```

## Quick Setup

### Method 1: Copy from existing Expanse output
```bash
# On SDSC Expanse
cd ~/lcgen
mkdir -p checkpoints/resume

# Copy from previous job
cp output/46136857-lcgen-baseline-metadata-xchannels/baseline_metadata_xchannels.pt \
   checkpoints/resume/model.pt
cp output/46136857-lcgen-baseline-metadata-xchannels/losses_per_epoch.npz \
   checkpoints/resume/losses_per_epoch.npz
```

### Method 2: Use helper script (on Expanse)
```bash
# On SDSC Expanse
cd ~/lcgen
chmod +x setup_resume_checkpoint.sh
./setup_resume_checkpoint.sh output/46136857-lcgen-baseline-metadata-xchannels
```

### Method 3: Copy from laptop
```bash
# On your laptop
cd ~/Documents/lc_ae
chmod +x copy_checkpoint_to_expanse.sh
./copy_checkpoint_to_expanse.sh output/slurm/46136857-lcgen-baseline-metadata-xchannels pvanlane
```

## Edit slurm.sh

Line 25:
```bash
RESUME_FROM="checkpoints/resume/model.pt"
```

Line 91 (or wherever --epochs is):
```bash
--epochs 100 \
```

## Submit Job

```bash
# On SDSC Expanse
cd ~/lcgen
sbatch slurm.sh
```

## Verify Resume Works

Check SLURM output log (`slurm_logs/lcgen-JOBID.out`):
```
============================================================
RESUMING FROM CHECKPOINT: checkpoint/model.pt
============================================================
✓ Loaded model state from checkpoint
✓ Loaded optimizer state from checkpoint
✓ Loaded previous losses: 20 epochs already completed
  Last train loss: 1.117498
  Last val loss: 1.173825

Resuming training from epoch 21 to epoch 100
============================================================
```

## Check Checkpoint Info

```bash
# On Expanse
cd ~/lcgen
python3 << 'EOF'
import numpy as np
losses = np.load('checkpoints/resume/losses_per_epoch.npz')
print(f"Epochs: {len(losses['train_loss'])}")
print(f"Last train loss: {losses['train_loss'][-1]:.6f}")
print(f"Last val loss: {losses['val_loss'][-1]:.6f}")
EOF
```

## Common Paths

| What | Path (relative to ~/lcgen) |
|------|---------------------------|
| Resume checkpoint | `checkpoints/resume/model.pt` |
| Resume losses | `checkpoints/resume/losses_per_epoch.npz` |
| Previous outputs | `output/JOBID-JOBNAME/` |
| SLURM logs | `slurm_logs/lcgen-JOBID.out` |

## Troubleshooting

**Path not found?**
- ✅ Use: `RESUME_FROM="checkpoints/resume/model.pt"`
- ❌ Don't use: `RESUME_FROM="$HOME/lcgen/checkpoints/resume/model.pt"`
- ❌ Don't use: Absolute paths

**Files missing?**
```bash
ls -lh ~/lcgen/checkpoints/resume/
# Should show model.pt and losses_per_epoch.npz
```

**Wrong architecture?**
Make sure resume training uses same flags as original:
- Same `--use_metadata`
- Same `--use_conv_channels`
- Same `--conv_encoder_type`
- Same `--hidden_size`
- Same `--direction`
