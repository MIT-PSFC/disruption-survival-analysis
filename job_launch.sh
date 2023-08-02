#!/bin/bash
#Input arguments are <number of times for script to run> <url of wandb run>
for i in {1.."$1"}
do
    sbatch job_instance.sh "$2"
done