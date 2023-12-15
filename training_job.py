# Look at the study for a given model
# Find the best trial so far
# Save its hyperparameters
# Train the model with best hyperparameters
# Save the model
# Calculate the metrics for the model (regular and bootstrapped)

import os
import sys

import dill

from disruption_survival_analysis.Experiments import Experiment
from disruption_survival_analysis.experiment_utils import load_experiment_config
from disruption_survival_analysis.manage_datasets import print_memory_usage

def main(device, dataset_path, model_type, alarm_type, metric, required_warning_time_ms, working_directory=None):

    if working_directory is not None:
        os.chdir(working_directory)
    
    print_memory_usage("Training before Creating Experiment")

    required_warning_time = float(required_warning_time_ms) / 1000
    config = load_experiment_config(device, dataset_path, model_type, alarm_type, metric, required_warning_time)

    # Create the experiment
    experiment = Experiment(config, 'test')
    sys.stdout.write("Experiment Trained Successfully!")
    print_memory_usage("Training after Creating Experiment")

    # Cache the basic metrics
    #experiment.get_critical_metrics_vs_thresholds()
    experiment.get_critical_metrics_vs_false_alarm_rates()
    sys.stdout.write("Experiment Metrics calculated")

    print_memory_usage("Training after Caching")

    # Make directory if it doesn't already exist
    directory_name = f"results/{device}/{dataset_path}/experiments"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    experiment_name = f"{model_type}_{alarm_type}_{metric}_{required_warning_time_ms}ms_experiment"
    with open(f"{directory_name}/{experiment_name}.pkl", 'wb') as f:
        dill.dump(experiment, f)

    print_memory_usage("Training after Saving to File")

    sys.stdout.write("Experiment saved to file")


if __name__ == "__main__":
    # Get the device, dataset path, model, alarm, metric, and min_warning_time from the command line
    device = sys.argv[1]
    dataset_path = sys.argv[2]
    model_type = sys.argv[3]
    alarm_type = sys.argv[4]
    metric = sys.argv[5]
    required_warning_time_ms = sys.argv[6]
    main(device, dataset_path, model_type, alarm_type, metric, required_warning_time_ms, sys.argv[7])