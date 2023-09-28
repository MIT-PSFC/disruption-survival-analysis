#!/bin/bash
#Launch 8 hyperparameter tuning workers for each sweep config file in the directory
project_name="test-multi-sweep-optuna"

# Activate Python environment
source /etc/profile
source ~/projects/disruption-survival-analysis/.venv/bin/activate

directory="synthetic/test"

# Get names of all sweep config files in directory
files=$(ls $directory | grep "sweep")

# Make a wandb agent for each sweep config file
for file in $files
do
    echo "Launching workers for $file"

    # Change the --job-name SBATCH argument in the job_instance.slurm file
    # to the name of the sweep config, minus .yaml
    sweep_id=$(echo $file | sed 's/.yaml//g')
    sed -i "s/--job-name=.*/--job-name=$sweep_id/g" job_instance.slurm

    # Launch the job
    ./job_launch.sh 7 $directory/$file
done