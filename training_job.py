# Look at the study for a given model
# Find the best trial so far
# Save its hyperparameters
# Train the model with best hyperparameters
# Save the model
# Calculate the metrics for the model (regular and bootstrapped)

import os
import sys

import dill
import optuna

from disruption_survival_analysis.Experiments import Experiment
from disruption_survival_analysis.experiment_utils import load_experiment_config

if __name__ == "__main__":
    # Get the device, dataset path, model, alarm, metric, and min_warning_time from the command line
    device = sys.argv[1]
    dataset_path = sys.argv[2]
    model_type = sys.argv[3]
    alarm_type = sys.argv[4]
    metric = sys.argv[5]
    required_warning_time_ms = sys.argv[6]

    # If an optional seventh argument is provided, change the working directory to that
    try:
        os.chdir(sys.argv[7])
    except:
        pass

    # Remove the previous config and model files if they exist
    model_name = f"{model_type}_{alarm_type}_{metric}_{required_warning_time_ms}ms"
    model_file = f"results/{device}/{dataset_path}/models/{model_name}.pkl"
    config_file = f"results/{device}/{dataset_path}/configs/{model_name}.yaml"
    try:
        os.remove(model_file)
    except FileNotFoundError:
        pass
    try:
        os.remove(config_file)
    except FileNotFoundError:
        pass

    config = load_experiment_config(device, dataset_path, model_type, alarm_type, metric, required_warning_time_ms)

    # Create the experiment
    experiment = Experiment(config, 'test')
    sys.stdout.write("Experiment Trained Successfully!")

    # Cache the basic metrics
    #experiment.get_critical_metrics_vs_thresholds()
    experiment.get_critical_metrics_vs_false_alarm_rates()
    sys.stdout.write("Experiment Metrics calculated")

    # Make directory if it doesn't already exist
    directory_name = f"results/{device}/{dataset_path}/experiments"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    experiment_name = f"{model_type}_{alarm_type}_{metric}_{required_warning_time_ms}ms_experiment"
    with open(f"{directory_name}/{experiment_name}.pkl", 'wb') as f:
        dill.dump(experiment, f)

    sys.stdout.write("Experiment saved to file")
