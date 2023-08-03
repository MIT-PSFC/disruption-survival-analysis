#!/bin/bash
#Input arguments are <number of times for script to run> <url of wandb run>
sbatch --array=0-$1 job_instance.slurm "$2"