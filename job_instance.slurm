#!/bin/bash
#SBATCH --job-name=cph_test_sweep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p sched_mit_hill
#SBATCH --time=00:00:10

source /etc/profile

# Load modules (if necessary)


# Activate Python environment
source ~/projects/disruption-survival-analysis/.venv/bin/activate

# Run instantiation of WandB agent with given url
# Runs a single training job then exits
wandb agent --count 1 "$1"

# Deactivate Python environment
deactivate