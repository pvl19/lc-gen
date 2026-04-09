# SDSC Expanse Folder Structure for Resume Training

## Recommended Directory Structure

```
$HOME/lcgen/                                    # PROJECT_DIR
├── data/                                        # Training data (already there)
│   ├── timeseries_x.h5
│   ├── star_sector_lc_formatted.pickle
│   └── TIC_cf_data.csv
│
├── src/                                         # Source code (already there)
│   └── lcgen/
│       ├── train_simple_rnn.py
│       └── ...
│
├── slurm.sh                                     # SLURM submission script
│
├── checkpoints/                                 # Generic checkpoint directory (NEW)
│   └── resume/                                  # Drop your resume files here
│       ├── model.pt                            # Copy your checkpoint here
│       ├── losses_per_epoch.npz                # Copy losses here
│       └── train_args.json                     # Optional: training config
│
└── output/                                      # Training outputs
    ├── 46136857-lcgen-baseline-metadata-xchannels/
    │   ├── baseline_metadata_xchannels.pt
    │   ├── losses_per_epoch.npz
    │   └── train_args.json
    ├── 46136813-lcgen-baseline-with-metadata/
    └── ...
```

## Setup Instructions

### 1. Create the Generic Checkpoint Directory

On SDSC Expanse:
```bash
ssh <username>@login.expanse.sdsc.edu
cd $HOME/lcgen
mkdir -p checkpoints/resume
```

### 2. Copy Files from Your Previous Training

You have two options:

#### Option A: Copy from existing output directory (on Expanse)
```bash
# On SDSC Expanse
cd $HOME/lcgen

# Copy from a previous SLURM job output
cp output/46136857-lcgen-baseline-metadata-xchannels/baseline_metadata_xchannels.pt \
   checkpoints/resume/model.pt

cp output/46136857-lcgen-baseline-metadata-xchannels/losses_per_epoch.npz \
   checkpoints/resume/losses_per_epoch.npz

cp output/46136857-lcgen-baseline-metadata-xchannels/train_args.json \
   checkpoints/resume/train_args.json
```

#### Option B: Copy from your laptop
```bash
# On your laptop
cd ~/Documents/lc_ae

# Create a temporary directory with the files you want to resume from
mkdir -p temp_resume
cp output/slurm/46136857-lcgen-baseline-metadata-xchannels/baseline_metadata_xchannels.pt \
   temp_resume/model.pt
cp output/slurm/46136857-lcgen-baseline-metadata-xchannels/losses_per_epoch.npz \
   temp_resume/losses_per_epoch.npz

# SCP to Expanse
scp temp_resume/* <username>@login.expanse.sdsc.edu:~/lcgen/checkpoints/resume/

# Clean up
rm -rf temp_resume
```

### 3. Update slurm.sh to Use Generic Path

Edit [slurm.sh](slurm.sh) line 25:
```bash
# Instead of:
# RESUME_FROM="output/46136857-lcgen-baseline-metadata-xchannels/baseline_metadata_xchannels.pt"

# Use:
RESUME_FROM="checkpoints/resume/model.pt"
```

### 4. Submit Your Job
```bash
sbatch slurm.sh
```

## Key Files Required for Resume

For resuming to work properly, you need **at minimum**:

1. **model.pt** (required)
   - Contains: model_state_dict, optimizer_state_dict, epoch count
   - The actual trained model weights

2. **losses_per_epoch.npz** (required)
   - Contains: train_loss array, val_loss array
   - Used to initialize epoch counter and continue loss history

3. **train_args.json** (optional but recommended)
   - Contains: all training arguments from previous run
   - Useful for reference to ensure you use matching architecture

## Example: Full Workflow

### Step 1: Train initial model (20 epochs)
```bash
# Edit slurm.sh:
# - Set RESUME_FROM="" (empty for fresh training)
# - Set --epochs 20

sbatch slurm.sh
# Job completes -> saves to output/46136857-lcgen-baseline-metadata-xchannels/
```

### Step 2: Copy to generic checkpoint directory
```bash
cd $HOME/lcgen
cp output/46136857-lcgen-baseline-metadata-xchannels/baseline_metadata_xchannels.pt \
   checkpoints/resume/model.pt
cp output/46136857-lcgen-baseline-metadata-xchannels/losses_per_epoch.npz \
   checkpoints/resume/losses_per_epoch.npz
```

### Step 3: Resume training (20 → 100 epochs)
```bash
# Edit slurm.sh:
# - Set RESUME_FROM="checkpoints/resume/model.pt"
# - Set --epochs 100

sbatch slurm.sh
# Job continues from epoch 21 → 100
```

### Step 4: Output
```bash
# New output saved to:
# output/46147892-lcgen-baseline-metadata-xchannels/
#   ├── baseline_metadata_xchannels.pt (model from epoch 100)
#   └── losses_per_epoch.npz (contains all 100 epochs)
```

## Tips

### Multiple Resume Checkpoints
If you want to keep multiple checkpoints for different experiments:

```
checkpoints/
├── resume/                    # Current resume (what slurm.sh uses)
│   ├── model.pt
│   └── losses_per_epoch.npz
├── backup_epoch20/            # Backup of 20-epoch model
│   ├── model.pt
│   └── losses_per_epoch.npz
└── backup_epoch50/            # Backup of 50-epoch model
    ├── model.pt
    └── losses_per_epoch.npz
```

Then you can easily swap which checkpoint to resume from:
```bash
# Resume from epoch 50 instead of current
cp checkpoints/backup_epoch50/* checkpoints/resume/
```

### Verify Files Are Correct
Before submitting, verify your checkpoint files:
```bash
# On Expanse
cd $HOME/lcgen/checkpoints/resume
ls -lh

# Should see:
# model.pt            (~200K for your current model)
# losses_per_epoch.npz (~few KB)
```

### Check What Epoch Your Checkpoint Is From
```bash
python3 << 'EOF'
import numpy as np
losses = np.load('checkpoints/resume/losses_per_epoch.npz')
print(f"Checkpoint has {len(losses['train_loss'])} epochs of training")
print(f"Last train loss: {losses['train_loss'][-1]:.6f}")
if 'val_loss' in losses:
    print(f"Last val loss: {losses['val_loss'][-1]:.6f}")
EOF
```

## Troubleshooting

**Problem**: "Checkpoint not found" error even though files exist

**Solution**: Make sure `RESUME_FROM` path is relative to `PROJECT_DIR`:
- ✅ Correct: `RESUME_FROM="checkpoints/resume/model.pt"`
- ❌ Wrong: `RESUME_FROM="$HOME/lcgen/checkpoints/resume/model.pt"`
- ❌ Wrong: `RESUME_FROM="/home/username/lcgen/checkpoints/resume/model.pt"`

---

**Problem**: Files get overwritten when copying from laptop

**Solution**: Use timestamped directories:
```bash
# On laptop
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
scp output/slurm/46136857-*/baseline_*.pt \
    username@login.expanse.sdsc.edu:~/lcgen/checkpoints/backup_${TIMESTAMP}/model.pt
```

---

**Problem**: Not sure which checkpoint to use

**Solution**: Check the SLURM output logs:
```bash
# On Expanse
cd $HOME/lcgen
grep "Saved final model" slurm_logs/lcgen-*.out | tail -5
```
