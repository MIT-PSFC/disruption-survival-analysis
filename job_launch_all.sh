#!/bin/bash
#Launch one wandb agent for each sweep config file in the directory
project_name="test-multi-sweep"

directory=$1

# Get names of all sweep config files in directory
files=$(ls $directory | grep "sweep")

# Make a wandb agent for each sweep config file
for file in $files
do
    # Launch the wandb agent
    wandb sweep --project $project_name $directory/$file
    # Get the sweep id from the output of the wandb sweep command
    sweep_id=$(echo $output | grep -oP '(?<=sweep\/).*(?=\/)')

    # Change the job name in the job_instance.slurm file
    sed -i "s/NAME/$sweep_id/g" job_instance.slurm

    # Launch the job
    sbatch --array=0-0 job_instance.slurm "$sweep_id"
done