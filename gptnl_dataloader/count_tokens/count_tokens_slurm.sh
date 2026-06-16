#!/bin/bash
#SBATCH --job-name="count-tokens"
#SBATCH --partition=genoa
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24

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

# pip install numpy pyarrow pandas
poetry run python $PATH_TO_DATALOADER/gptnl_dataloader/count_tokens/count_tokens.py
