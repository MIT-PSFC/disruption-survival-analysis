#!/bin/bash
#SBATCH --job-name=cph_hyst_auroc_100ms_sweep
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -p sched_mit_psfc_r8
#SBATCH --time=08:00:00
#SBATCH -o ./slurm/slurm-%j.out

source /etc/profile

# Load modules (if necessary)

# Activate Python environment
source ~/projects/disruption-survival-analysis/.venv/bin/activate

# Run instantiation of optuna hyperparameter tuning for a given sweep
# Runs n training jobs in parallel
# srun wandb agent --count 256 "$1"
python optuna_job.py "$1"

# Deactivate Python environment
deactivate