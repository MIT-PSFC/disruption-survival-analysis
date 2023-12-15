#!/bin/bash
# 1. device
# 2. dataset path
# 3. model type
# 4. alarm type
# 5. metric
# 6. min warning time [ms] string
# 7. number of bootstrap slices
# 8. working directory
# 9. memory per CPU

cat <<EoF
#!/bin/bash
#SBATCH --job-name=slice-job-$1-$2-$3-$4-$5-$6
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=$9MB
#SBATCH -p sched_mit_psfc_r8
#SBATCH --time=08:00:00
#SBATCH -o ./slurm/slurm-%j.out

source /etc/profile

# Load modules (if necessary)

# Activate Python environment
source ~/projects/disruption-survival-analysis/.venv/bin/activate

# Run Python script
python bootstrap_coalesce.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"

# Deactivate Python environment
deactivate

EoF