#!/usr/bin/env bash
# SLURM training script with checkpoint resume support
#
# To resume from a previous checkpoint:
#   1. Set RESUME_FROM variable below (line 26) to point to your checkpoint
#   2. Update --epochs to your new target (e.g., 50 or 100)
#   3. Training will continue from where it left off
#
#SBATCH --job-name=lcgen-base-single-k-cont
#### Change account below
#SBATCH --account=csd902
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/lcgen-%j.out
#SBATCH --error=slurm_logs/lcgen-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pvanlane@ucsd.edu

#### Define paths
PROJECT_DIR="$HOME/lcgen"
SCRATCH_DIR="/scratch/$USER/job_$SLURM_JOBID"
OUTPUT_DIR="$PROJECT_DIR/output/$SLURM_JOBID-$SLURM_JOB_NAME"

#### OPTIONAL: Set to resume training from a previous checkpoint
#### Recommended: Use checkpoints/resume/model.pt (see EXPANSE_FOLDER_STRUCTURE.md)
#### Example: RESUME_FROM="checkpoints/resume/model.pt"
#### Or direct path: RESUME_FROM="output/46136857-lcgen-baseline-metadata-xchannels/baseline_metadata_xchannels.pt"
#### Leave empty ("") for fresh training
RESUME_FROM="checkpoints/resume/model.pt"

declare -xr SINGULARITY_MODULE='singularitypro/3.11'

module purge
module load "${SINGULARITY_MODULE}"
module list

echo "========================================"
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Project directory: $PROJECT_DIR"
echo "Scratch directory: $SCRATCH_DIR"
echo "========================================"

#### Copy data to scratch for faster I/O
mkdir -p $SCRATCH_DIR
cp -r $PROJECT_DIR/data $SCRATCH_DIR/
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

#### mkdir output directory on scratch
mkdir -p $SCRATCH_DIR/output

#### Change to scratch directory
cd $SCRATCH_DIR

#### Set PYTHONPATH so imports work
export PYTHONPATH="$SCRATCH_DIR/src:$PYTHONPATH"

echo "Installing additional dependencies..."
echo "========================================"

singularity exec --bind /expanse,/scratch --nv \
    /cm/shared/apps/containers/singularity/pytorch/pytorch_2.2.1-openmpi-4.1.6-mofed-5.8-2.0.3.0-cuda-12.1.1-ubuntu-22.04.4-x86_64-20240412.sif \
    pip install --user zuko

echo "Running training..."
echo "========================================"

#### Build resume flag if checkpoint specified
RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    CHECKPOINT_NAME=$(basename "$RESUME_FROM")
    RESUME_FLAG="--resume_from checkpoint/$CHECKPOINT_NAME"
fi

#### Run training with singularity container (has PyTorch + CUDA)
#### Use python3 -u for unbuffered output so logs appear in real-time
time -p singularity exec --bind /expanse,/scratch --nv \
    /cm/shared/apps/containers/singularity/pytorch/pytorch_2.2.1-openmpi-4.1.6-mofed-5.8-2.0.3.0-cuda-12.1.1-ubuntu-22.04.4-x86_64-20240412.sif \
    python3 -u src/lcgen/train_simple_rnn.py \
        --input data/timeseries_x.h5 \
        --output_dir output \
        --random_seed 19 \
        --direction bi \
        --max_length 2048 \
        --num_samples 8192 \
        --batch_size 256 \
        --hidden_size 64 \
        --output_name model.pt \
        --epochs 56 \
        --lr 1e-3 \
        --min_size 5 \
        --max_size 256 \
        --mask_portion 0.4 \
        --use_flow \
        --mode parallel \
        --val_split 0.1 \
        --val_k_values 1,2,4,8 \
        --K 128 \
        --k_spacing log \
        --patience 10 \
        --min_delta 0.0 \
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
