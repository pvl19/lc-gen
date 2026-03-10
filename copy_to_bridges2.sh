#!/usr/bin/env bash
# Copy updated source code, data, and SLURM script to PSC Bridges-2.
# Usage: bash copy_to_bridges2.sh
#
# Prerequisites: an SSH host alias 'bridges2' configured in ~/.ssh/config, e.g.:
#   Host bridges2
#       HostName bridges2.psc.edu
#       User <your_username>
#       IdentityFile ~/.ssh/id_rsa

REMOTE="bridges2"
REMOTE_DIR="~/lcgen"   # ~ is left unquoted so the remote shell expands it

echo "Copying source code..."
rsync -avz --progress \
    src/lcgen/train_simple_rnn.py \
    $REMOTE:$REMOTE_DIR/src/lcgen/

rsync -avz --progress \
    src/lcgen/models/simple_min_gru.py \
    src/lcgen/models/TimeSeriesDataset.py \
    src/lcgen/models/MetadataAgePredictor.py \
    src/lcgen/models/TransformerAgePredictor.py \
    src/lcgen/models/lightweight_conv.py \
    src/lcgen/models/conv_models.py \
    src/lcgen/models/rnn.py \
    $REMOTE:$REMOTE_DIR/src/lcgen/models/

rsync -avz --progress \
    src/lcgen/utils/loss.py \
    src/lcgen/utils/mask.py \
    src/lcgen/utils/run_log.py \
    src/lcgen/utils/trunc_data.py \
    src/lcgen/utils/conv_data_prep.py \
    $REMOTE:$REMOTE_DIR/src/lcgen/utils/

echo ""
echo "Copying SLURM script..."
rsync -avz --progress \
    slurm_bridges2.sh \
    $REMOTE:$REMOTE_DIR/

echo ""
echo "Copying data (this may take a while – timeseries_x.h5 is large)..."
rsync -avz --progress \
    data/timeseries_x.h5 \
    $REMOTE:$REMOTE_DIR/data/

echo ""
echo "========================================"
echo "Done! Files copied to $REMOTE:$REMOTE_DIR"
echo ""
echo "Next steps on Bridges-2:"
echo "  ssh $REMOTE"
echo "  cd ~/lcgen"
echo "  chmod +x slurm_bridges2.sh"
echo "  sbatch slurm_bridges2.sh"
echo "========================================"
