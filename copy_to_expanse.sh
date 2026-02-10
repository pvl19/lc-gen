# Copy everything at once
rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude '.DS_Store' \
  --exclude '*.ipynb_checkpoints' \
  /Users/philvanlane/Documents/lc_ae/slurm.sh \
  /Users/philvanlane/Documents/lc_ae/src/ \
  /Users/philvanlane/Documents/lc_ae/data/timeseries_x.h5 \
  expanse:~/lcgen/


# Just code
rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude '.DS_Store' \
  --exclude '*.ipynb_checkpoints' \
  /Users/philvanlane/Documents/lc_ae/src \
  expanse:~/lcgen/


# Re-copy the updated source code
rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
  /Users/philvanlane/Documents/lc_ae/src/ expanse:~/lcgen/src/

# Just slurm
rsync -avz --progress \
  /Users/philvanlane/Documents/lc_ae/slurm.sh \
  expanse:~/lcgen/

# Copy model results back
rsync -avz --progress \
  expanse:~/lcgen/output/46270143-lcgen-base-single-k \
  /Users/philvanlane/Documents/lc_ae/output/slurm/

# Copy checkpoints over
rsync -avz --progress \
  /Users/philvanlane/Documents/lc_ae/checkpoints \
  expanse:~/lcgen/