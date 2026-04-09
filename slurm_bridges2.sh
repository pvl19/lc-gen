#!/usr/bin/env bash
# SLURM training script for PSC Bridges-2
# Documentation: https://www.psc.edu/resources/bridges-2/user-guide/
#
# To resume from a previous checkpoint:
#   1. Set RESUME_FROM variable below to point to your checkpoint
#   2. Update --epochs to your new target (e.g., 50 or 100)
#   3. Training will continue from where it left off
#
#SBATCH --job-name=lcgen-final-parallel-v2
#### Change account to your allocation (e.g., abc123p)
#SBATCH --account=phy260003p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# cpus-per-task = nproc_per_node * (num_workers + 1)
# GPU-shared (4 GPUs): 4 * (4 + 1) = 20
#SBATCH --cpus-per-task=20
#SBATCH --gpus=v100-32:4
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/lcgen-%j.out
#SBATCH --error=slurm_logs/lcgen-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pvanlane@ucsd.edu

#### Define paths
PROJECT_DIR="$HOME/lcgen"
# Use local SSD scratch for fast I/O during job
SCRATCH_DIR="$LOCAL/job_$SLURM_JOBID"
OUTPUT_DIR="$PROJECT_DIR/output/$SLURM_JOBID-$SLURM_JOB_NAME"

#### OPTIONAL: Set to resume training from a previous checkpoint
#### Example: RESUME_FROM="checkpoints/resume/model.pt"
#### Leave empty ("") for fresh training
RESUME_FROM="checkpoints/resume/model.pt"

# Define container
CONTAINER="/ocean/containers/ngc/pytorch/delete/pytorch_24.11-py3.sif"

echo "========================================"
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Project directory: $PROJECT_DIR"
echo "Scratch directory: $SCRATCH_DIR"
echo "Container: $CONTAINER"
echo "========================================"

#### Copy data to local scratch for faster I/O
mkdir -p $SCRATCH_DIR
echo "Copying data to local scratch..."
cp -r $PROJECT_DIR/final_pretrain $SCRATCH_DIR/
cp -r $PROJECT_DIR/src $SCRATCH_DIR/

#### Copy checkpoint if resuming from previous training
if [ -n "$RESUME_FROM" ]; then
    CHECKPOINT_DIR=$(dirname "$PROJECT_DIR/$RESUME_FROM")
    CHECKPOINT_NAME=$(basename "$RESUME_FROM")
    mkdir -p $SCRATCH_DIR/checkpoint
    cp "$CHECKPOINT_DIR"/* $SCRATCH_DIR/checkpoint/ 2>/dev/null || true
    echo "Resuming from checkpoint: $RESUME_FROM"
else
    echo "Starting fresh training (no checkpoint specified)"
fi

#### Create output directory on scratch
mkdir -p $SCRATCH_DIR/output

#### Change to scratch directory
cd $SCRATCH_DIR

#### Install dependencies (if needed)
echo "Installing dependencies..."
singularity exec --nv --bind /ocean,$LOCAL,$HOME \
    $CONTAINER pip install --user zuko h5py 2>/dev/null || true

#### Verify setup
echo "Verifying PyTorch + CUDA..."
singularity exec --nv --bind /ocean,$LOCAL,$HOME \
    $CONTAINER python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo "========================================"
echo "Starting training..."
echo "========================================"

#### Build resume flag if checkpoint specified
RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    CHECKPOINT_NAME=$(basename "$RESUME_FROM")
    RESUME_FLAG="--resume_from checkpoint/$CHECKPOINT_NAME"
fi

#### Run training with Singularity container (4-GPU DDP via torchrun)
time -p singularity exec --nv --bind /ocean,$LOCAL,$HOME \
    $CONTAINER torchrun \
    --nproc_per_node=4 \
    --standalone \
    src/lcgen/train_simple_rnn.py \
    --input final_pretrain/timeseries_pretrain.h5 final_pretrain/timeseries_exop_hosts.h5 \
    --output_dir output \
    --random_seed 19 \
    --direction bi \
    --batch_size 64 \
    --hidden_size 64 \
    --output_name model.pt \
    --epochs 50 \
    --lr 1e-3 \
    --min_size 5 \
    --max_size 720 \
    --mask_portion 0.4 \
    --use_flow \
    --use_metadata \
    --mode parallel \
    --val_split 0.1 \
    --val_k_values 1,2,4,8 \
    --num_workers 4 \
    --K 720 \
    --k_spacing log \
    --patience 10 \
    --min_delta 0.0 \
    --save_every 1 \
    --checkpoint_copy_dir $PROJECT_DIR/checkpoints/resume \
    $RESUME_FLAG

echo "========================================"
echo "Training complete. Copying results back..."
echo "========================================"

#### Copy output back to project directory
mkdir -p $OUTPUT_DIR
cp -r $SCRATCH_DIR/output/* $OUTPUT_DIR/

#### List what was saved
echo "Files saved to $OUTPUT_DIR:"
ls -la $OUTPUT_DIR/

echo "========================================"
echo "End time: $(date)"
echo "========================================"