#!/usr/bin/env bash
# Test script for PSC Bridges-2 - run in interactive session
#
# First, get an interactive GPU session:
#   interact -p GPU-shared --gres=gpu:v100-32:1 -t 00:30:00 -A phy260003p
#
# Then run this script:
#   cd $PROJECT/lcgen
#   ./test_bridges2.sh

echo "========================================"
echo "Testing Bridges-2 Setup"
echo "========================================"

module load python

# Define paths (singularity is available by default on Bridges-2)
PROJECT_DIR="$HOME/lcgen"
# Use pre-staged NGC PyTorch container (no need to pull)
CONTAINER="/ocean/containers/ngc/pytorch/pytorch_24.11-py3.sif"

echo ""
echo "1. Checking paths..."
echo "   PROJECT: $PROJECT"
echo "   PROJECT_DIR: $PROJECT_DIR"
echo "   CONTAINER: $CONTAINER"

if [ ! -f "$CONTAINER" ]; then
    echo "   ERROR: Container not found at $CONTAINER"
    echo "   Available containers:"
    ls -la /ocean/containers/ngc/pytorch/ 2>/dev/null || echo "     Cannot list /ocean/containers/ngc/pytorch/"
    exit 1
fi
echo "   Container found!"

echo ""
echo "2. Checking data files..."
if [ -f "$PROJECT_DIR/data/timeseries_x.h5" ]; then
    echo "   timeseries_x.h5 found!"
    ls -lh "$PROJECT_DIR/data/timeseries_x.h5"
else
    echo "   WARNING: timeseries_x.h5 not found"
fi

if [ -f "$PROJECT_DIR/data/timeseries.h5" ]; then
    echo "   timeseries.h5 found!"
    ls -lh "$PROJECT_DIR/data/timeseries.h5"
else
    echo "   WARNING: timeseries.h5 not found"
fi

echo ""
echo "3. Checking source files..."
if [ -f "$PROJECT_DIR/src/lcgen/train_simple_rnn.py" ]; then
    echo "   train_simple_rnn.py found!"
else
    echo "   ERROR: train_simple_rnn.py not found"
    exit 1
fi

echo ""
echo "4. Testing container + PyTorch + CUDA..."
singularity exec --nv --bind /ocean \
    $CONTAINER python -c "
import torch
print(f'   PyTorch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('   ERROR: CUDA not available!')
    exit(1)
"

echo ""
echo "5. Installing dependencies (if needed)..."
singularity exec --nv --bind /ocean \
    $CONTAINER pip install --user zuko h5py 2>/dev/null

echo ""
echo "6. Testing imports..."
cd $PROJECT_DIR
singularity exec --nv --bind /ocean \
    $CONTAINER python -c "
import sys
sys.path.insert(0, 'src')

print('   Importing torch...', end=' ')
import torch
print('OK')

print('   Importing zuko...', end=' ')
import zuko
print('OK')

print('   Importing h5py...', end=' ')
import h5py
print('OK')

print('   Importing model...', end=' ')
from lcgen.models.simple_min_gru import BiDirectionalMinGRU
print('OK')

print('   Importing loss...', end=' ')
from lcgen.utils.loss import bounded_horizon_future_nll
print('OK')

print('   Importing dataset...', end=' ')
from lcgen.models.TimeSeriesDataset import TimeSeriesDataset
print('OK')
"

echo ""
echo "7. Testing data loading (quick)..."
singularity exec --nv --bind /ocean \
    $CONTAINER python -c "
import sys
sys.path.insert(0, 'src')
import h5py

h5_path = 'data/timeseries_x.h5'
print(f'   Loading {h5_path}...')
with h5py.File(h5_path, 'r') as f:
    print(f'   Keys: {list(f.keys())}')
    print(f'   Flux shape: {f[\"flux\"].shape}')
    print(f'   Number of samples: {len(f[\"flux\"])}')
print('   Data loading OK!')
"

echo ""
echo "8. Testing model creation..."
singularity exec --nv --bind /ocean \
    $CONTAINER python -c "
import sys
sys.path.insert(0, 'src')
import torch
from lcgen.models.simple_min_gru import BiDirectionalMinGRU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'   Device: {device}')

model = BiDirectionalMinGRU(
    hidden_size=64,
    direction='bi',
    mode='parallel',
    use_flow=True
).to(device)
print(f'   Model created successfully!')
print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
"

echo ""
echo "========================================"
echo "All tests passed! Ready to run training."
echo "========================================"
echo ""
echo "Submit your job with:"
echo "  sbatch slurm_bridges2.sh"
