#!/bin/bash
#SBATCH --job-name=bootstrap_job
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=24576MB
#SBATCH -p sched_mit_psfc_r8
#SBATCH --time=08:00:00
#SBATCH -o ./slurm/slurm-%j.out

source /etc/profile

# Load modules (if necessary)

# Activate Python environment
source ~/projects/disruption-survival-analysis/.venv/bin/activate

# Run Python script
python bootstrap_job.py "$1" "$2" "$3" "$4" "$5" "$6"

# Deactivate Python environment
deactivate