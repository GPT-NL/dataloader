#!/bin/bash
#SBATCH --job-name="bucketing"
#SBATCH --partition=genoa
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72

# ---------------------------------------------------------------
# STEP 0: Load necessary modules

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

PATH_TO_DATALOADER=${HOME}/dataloader
VENV_ACTIVATE=${PATH_TO_DATALOADER}/.venv/bin/activate

if [ -f "$VENV_ACTIVATE" ]; then
    echo "Activating virtual environment ($VENV_ACTIVATE)..."
    source $VENV_ACTIVATE
else
    echo "Virtual environment activation script not found: $VENV_ACTIVATE". Run 'poetry install' to create the virtual environment.
    exit 1
fi

# ---------------------------------------------------------------
# STEP 1: SYMLINK curated sample

source ${PATH_TO_DATALOADER}/.env
source config.env

# Make a symlink instead of copying
echo "Creating symlink for curated sample from $CURATED_OUTPUT_DIR to $COPY_CURATED_OUTPUT_DIR"
mkdir -p $COPY_CURATED_OUTPUT_DIR
ln -s $CURATED_OUTPUT_DIR $COPY_CURATED_OUTPUT_DIR  
echo "Creating symlink complete."

# ---------------------------------------------------------------
# STEP 2: Run tagging
echo "Starting tagging..."
poetry run python -u -m gptnl_dataloader.bucketing.tag --batch-size 30

# ---------------------------------------------------------------
# STEP 3: Run bucketing
echo "Starting bucketing..."
poetry run python -u -m gptnl_dataloader.bucketing.bucketing --batch-size 20 --max-size-mb 500

# ---------------------------------------------------------------
# STEP 4: Post-bucketing check
echo "Starting post-bucketing check..."
poetry run python -u -m gptnl_dataloader.bucketing.post_bucketing_check
