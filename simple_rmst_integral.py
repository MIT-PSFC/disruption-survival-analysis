import os
import sys

import dill

from disruption_survival_analysis.Experiments import Experiment
from disruption_survival_analysis.manage_datasets import print_memory_usage
from disruption_survival_analysis.experiment_utils import load_experiment_config

def main(device, dataset_path, model_type, alarm_type, metric, required_warning_time_ms, working_directory=None):
    # If an optional sixth argument is provided, change the working directory to that
    if working_directory is not None:
        os.chdir(working_directory)

    required_warning_time = int(required_warning_time_ms)/1000
    config = load_experiment_config(device, dataset_path, model_type, alarm_type, metric, required_warning_time)

    print_memory_usage("Simple RMST Before Creating Experiment")

    # Create the experiment
    experiment = Experiment(config, 'test')
    sys.stdout.write("Created experiment\n")

    print_memory_usage("Simple RMST After Creating Experiment")

    disruptive_rmst_diffs, non_disruptive_rmst_diffs = experiment.get_simple_rmst_integrals()

    print_memory_usage("Simple RMST After Getting Results")

    rmst_dir = f"{model_type}_{alarm_type}_{metric}_{required_warning_time_ms}ms"

    # Make directory if it doesn't already exist
    directory_name = f"results/{device}/{dataset_path}/simple_rmst/{rmst_dir}"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    results = {}
    results['disruptive_rmst_diffs'] = disruptive_rmst_diffs
    results['non_disruptive_rmst_diffs'] = non_disruptive_rmst_diffs

    with open(f"{directory_name}/all_rmst_results.pkl", 'wb') as f:
        dill.dump(results, f)

    sys.stdout.write("Saved all RMST integrals")

