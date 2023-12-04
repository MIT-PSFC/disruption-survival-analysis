#!/bin/bash
#SBATCH --job-name=tuning_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH -p sched_mit_psfc_r8
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8000MB
#SBATCH -o ./slurm/slurm-%j.out

source /etc/profile

# Load modules (if necessary)

# Activate Python environment
source ~/projects/disruption-survival-analysis/.venv/bin/activate

# Run 4 training jobs in parallel
srun --exclusive --ntasks=1 --mem-per-cpu 8000MB python optuna_job.py "$1" &
srun --exclusive --ntasks=1 --mem-per-cpu 8000MB python optuna_job.py "$1" &
srun --exclusive --ntasks=1 --mem-per-cpu 8000MB python optuna_job.py "$1" &
srun --exclusive --ntasks=1 --mem-per-cpu 8000MB python optuna_job.py "$1" &
wait

# Deactivate Python environment
deactivate