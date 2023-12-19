#!/bin/bash
# 1. number of job instances to launch
# 2. sweep config path
# 3. working directory
# 4. memory per cpu

cat <<EoF
#!/bin/bash
#SBATCH --job-name=tuning-job-$2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$1
#SBATCH --cpus-per-task=1
#SBATCH -p sched_mit_psfc_r8
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=$4MB
#SBATCH -o ./slurm/slurm-%j-tuning.out

source /etc/profile

# Load modules (if necessary)

# Activate Python environment
source ~/projects/disruption-survival-analysis/.venv/bin/activate
EoF

for i in $(seq 1 $1)
do
    echo srun --exclusive --ntasks=1 --cpus-per-task=1 --mem-per-cpu $4MB python optuna_job.py "$2" "$3" "&"
done

echo wait

echo deactivate