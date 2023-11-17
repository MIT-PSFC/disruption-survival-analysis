#!/bin/bash
#SBATCH --job-name=tuning_job
#SBATCH --ntasks 16
#SBATCH --cpus-per-task=1
#SBATCH -p sched_mit_psfc_r8
#SBATCH --time=08:00:00
#SBATCH -o ./slurm/slurm-%j.out

source /etc/profile

# Load modules (if necessary)

# Activate Python environment
source ~/projects/disruption-survival-analysis/.venv/bin/activate

# Run 10 training jobs in parallel
srun --ntasks=1 python optuna_job.py "$1" &
srun --ntasks=1 python optuna_job.py "$1" &
srun --ntasks=1 python optuna_job.py "$1" &
srun --ntasks=1 python optuna_job.py "$1" &
srun --ntasks=1 python optuna_job.py "$1" &
srun --ntasks=1 python optuna_job.py "$1" &
srun --ntasks=1 python optuna_job.py "$1" &
srun --ntasks=1 python optuna_job.py "$1" &
srun --ntasks=1 python optuna_job.py "$1" &
srun --ntasks=1 python optuna_job.py "$1" &
wait

# Deactivate Python environment
deactivate