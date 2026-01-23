#!/usr/bin/env bash
#SBATCH --job-name=lcgen-train-01
#### Change account below
#SBATCH --account=csd902
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=lcgen-%j.out
#SBATCH --error=lcgen-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pvanlane@ucsd.edu

#### Define paths
PROJECT_DIR="$HOME/lcgen"
SCRATCH_DIR="/scratch/$USER/job_$SLURM_JOBID"
OUTPUT_DIR="$PROJECT_DIR/output"

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

#### Create output directory on scratch
mkdir -p $SCRATCH_DIR/output/models
mkdir -p $SCRATCH_DIR/output/logs
mkdir -p $SCRATCH_DIR/output/training_loss

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

#### Run training with singularity container (has PyTorch + CUDA)
#### Use python3 -u for unbuffered output so logs appear in real-time
time -p singularity exec --bind /expanse,/scratch --nv \
    /cm/shared/apps/containers/singularity/pytorch/pytorch_2.2.1-openmpi-4.1.6-mofed-5.8-2.0.3.0-cuda-12.1.1-ubuntu-22.04.4-x86_64-20240412.sif \
    python3 -u src/lcgen/train_simple_rnn.py \
        --input data/timeseries_x.h5 \
        --output_dir output \
        --random_seed 19 \
        --direction bi \
        --max_length 16384 \
        --num_samples 10000 \
        --batch_size 2048 \
        --output_name smoke_test_meta.pt \
        --epochs 2 \
        --lr 1e-3 \
        --min_size 5 \
        --max_size 1024 \
        --mask_portion 0.4 \
        --use_flow \
        --use_metadata \
        --mode parallel \
        --K 8192 \
        --k_spacing fibonacci

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
