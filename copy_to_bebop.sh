#!/usr/bin/env bash
# Copy necessary files to bebop server for training

REMOTE="bootes"
REMOTE_DIR="~/lcgen"

echo "Creating remote directory structure..."
ssh $REMOTE "mkdir -p $REMOTE_DIR/src/lcgen/models $REMOTE_DIR/src/lcgen/utils $REMOTE_DIR/data $REMOTE_DIR/output"

echo "Copying source code..."
rsync -avz --progress \
    src/lcgen/*.py \
    $REMOTE:$REMOTE_DIR/src/lcgen/

rsync -avz --progress \
    src/lcgen/models/*.py \
    $REMOTE:$REMOTE_DIR/src/lcgen/models/

rsync -avz --progress \
    src/lcgen/utils/*.py \
    $REMOTE:$REMOTE_DIR/src/lcgen/utils/

echo "Copying training script..."
rsync -avz --progress \
    train_model.sh \
    $REMOTE:$REMOTE_DIR/

echo "Copying data file (this may take a while)..."
rsync -avz --progress \
    data/timeseries_x.h5 \
    $REMOTE:$REMOTE_DIR/data/

echo "Copying requirements..."
rsync -avz --progress \
    requirements.txt \
    $REMOTE:$REMOTE_DIR/ 2>/dev/null || echo "No requirements.txt found, skipping"

echo "========================================"
echo "Done! Files copied to $REMOTE:$REMOTE_DIR"
echo ""
echo "Next steps on bebop:"
echo "  1. ssh $REMOTE"
echo "  2. cd $REMOTE_DIR"
echo "  3. pip install --user zuko  # if not already installed"
echo "  4. chmod +x train_model.sh"
echo "  5. nohup ./train_model.sh > training.log 2>&1 &"
echo "========================================"
