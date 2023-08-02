#!/bin/bash
#SBATCH --job-name=cph_test_sweep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p sched_mit_psfc_gpu_r8
#SBATCH -t 0-01:00
#SBATCH --array

source "etc/profile"

# Load modules (if necessary)


# Activate Python environment
source "~/projects/disruption-survival-analysis/.venv/Scripts/activate"

# Run instantiation of WandB agent with given url
wandb agent "$1"

# Deactivate Python environment
deactivate