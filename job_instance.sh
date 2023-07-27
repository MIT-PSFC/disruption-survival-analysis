#!/bin/bash

source "/c/users/zkeith/documents/Risk-Aware Frameworks/disruption-survival-analysis/"

# Load modules (if necessary)


# Activate Python environment
source "/c/users/zkeith/documents/Risk-Aware Frameworks/disruption-survival-analysis/.venv/Scripts/activate"
which python

# Run instantiation of WandB agent with given url
wandb agent "$1"

# Deactivate Python environment
deactivate