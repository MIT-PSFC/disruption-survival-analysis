#!/bin/bash
# 1. device
# 2. dataset path
# 3. model type
# 4. alarm type
# 5. metric
# 6. min warning time [ms] string
# 7. working directory
# 8. memory per CPU

#!/bin/bash
cat <<EoF
########## Begin Slurm header ##########
#SBATCH --job-name=training_job
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=$8MB
#SBATCH -p sched_mit_psfc_r8
#SBATCH --time=08:00:00
#SBATCH -o ./slurm/slurm-%j.out
########### End Slurm header ##########


source /etc/profile

# Load modules (if necessary)

# Activate Python environment
source ~/projects/disruption-survival-analysis/.venv/bin/activate

# Run training job script
python training_job.py "$1" "$2" "$3" "$4" "$5" "$6" "$7"

# Deactivate Python environment
deactivate

EoF