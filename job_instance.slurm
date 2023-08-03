#!/bin/bash
#SBATCH --job-name=cph-tibs
#SBATCH -N 1 
#SBATCH -n 16
#SBATCH --cpus-per-task=1
#SBATCH -p sched_mit_psfc_r8
#SBATCH --time=01:00:00
#SBATCH -o ./slurm/slurm-%j.out

source /etc/profile

# Load modules (if necessary)

# Activate Python environment
source ~/projects/disruption-survival-analysis/.venv/bin/activate

# Run instantiation of WandB agent with given url
# Runs n training jobs in parallel
srun wandb agent --count 1 "$1"


# Deactivate Python environment
deactivate