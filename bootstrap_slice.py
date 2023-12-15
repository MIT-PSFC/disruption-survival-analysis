import os
import sys

import dill

from disruption_survival_analysis.Experiments import Experiment
from disruption_survival_analysis.manage_datasets import print_memory_usage
from disruption_survival_analysis.experiment_utils import load_experiment_config

def main(device, dataset_path, model_type, alarm_type, metric, required_warning_time_ms, bootstrap_number, working_directory=None):
    # If an optional seventh argument is provided, change the working directory to that
    if working_directory is not None:
        os.chdir(working_directory)

    required_warning_time = int(required_warning_time_ms)/1000
    config = load_experiment_config(device, dataset_path, model_type, alarm_type, metric, required_warning_time)

    print_memory_usage("Slice Before Creating Experiment")

    # Create the experiment
    experiment = Experiment(config, 'test')
    sys.stdout.write("Created experiment\n")

    print_memory_usage("Slice After Creating Experiment")

    false_alarm_rates, true_alarm_metrics, warning_time_metrics = experiment.get_critical_metrics_vs_false_alarm_rates(None, None, bootstrap_number, ['avg', 'iqm'])

    print_memory_usage("Slice After Getting Results")

    # Make directory if it doesn't already exist
    directory_name = f"results/{device}/{dataset_path}/bootstraps"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    slice_name = f"{model_type}_{alarm_type}_{metric}_{required_warning_time_ms}ms_slice_{bootstrap_number}"

    with open(f"{directory_name}/{slice_name}.pkl", 'wb') as f:
        dill.dump(false_alarm_rates, true_alarm_metrics, warning_time_metrics, f)

    sys.stdout.write("Saved bootstrapped metrics")


if __name__ == "__main__":
    # Get the device, dataset path, model, alarm, metric, min_warning_time, and working directory from the command line
    device = sys.argv[1]
    dataset_path = sys.argv[2]
    model_type = sys.argv[3]
    alarm_type = sys.argv[4]
    metric = sys.argv[5]
    required_warning_time_ms = sys.argv[6]
    bootstrap_number = int(sys.argv[7])
    try:
        working_directory = sys.argv[8]
    except:
        working_directory = None

    main(device, dataset_path, model_type, alarm_type, metric, required_warning_time_ms, bootstrap_number, working_directory)