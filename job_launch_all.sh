#!/bin/bash
#Launch one wandb agent for each sweep config file in the directory
project_name="test-multi-sweep"

# Activate Python environment
source /etc/profile
source ~/projects/disruption-survival-analysis/.venv/bin/activate

#wandb --version

directory="models/synthetic/synthetic100"

# Get names of all sweep config files in directory
files=$(ls $directory | grep "sweep")

# Make a wandb agent for each sweep config file
for file in $files
do
    echo "Launching agent for $file"
    # Launch the wandb agent
    # Launching the sweep produces 4 lines of text, the last of which contains the sweep url
    # We want to obtain this url
    # The url appears after the string "wandb: Run sweep agent with: "
    sweep_launch_output="$(wandb sweep --project $project_name $directory/$file 2>&1)"
    last_line=$(echo "$sweep_launch_output" | tail -n 1)
    echo "$last_line"
    sweep_url=$(echo $last_line | sed 's/wandb: Run sweep agent with: wandb agent //g')

    echo "Sweep url: $sweep_url"

    # Change the --job-name SBATCH argument in the job_instance.slurm file
    # to the name of the sweep config, minus .yaml
    sweep_id=$(echo $file | sed 's/.yaml//g')
    sed -i "s/--job-name=.*/--job-name=$sweep_id/g" job_instance.slurm

    # Launch the job
    ./job_launch.sh 0 $sweep_url
done