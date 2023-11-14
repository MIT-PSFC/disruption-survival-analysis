#!/bin/bash
# 1. device
# 2. dataset path
# 3. model type
# 4. alarm type
# 5. metric
# 6. min warning time [ms] string
sbatch bootstrap_instance.slurm "$1" "$2" "$3" "$4" "$5" "$6"