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

if __name__ == "__main__":
    # Get the device, dataset path, model, alarm, metric, and min_warning_time from the command line
    device = sys.argv[1]
    dataset_path = sys.argv[2]
    model_type = sys.argv[3]
    alarm_type = sys.argv[4]
    metric = sys.argv[5]
    required_warning_time_ms = sys.argv[6]
    
    study_name = f"{model_type}_{alarm_type}_{metric}_{required_warning_time_ms}ms_study"

    # Load the study
    study_path = f"models/{device}/{dataset_path}/studies/{study_name}.db"

    # Check if the database file exists
    with open(study_path, "r") as f:
        pass

    lock_obj = optuna.storages.JournalFileOpenLock(study_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(study_path, lock_obj=lock_obj)
    )

    # Get the best trial from the study (expects there to be only one study in the database)
    study = optuna.load_study(study_name=None, storage=storage)

    hyperparameters = study.best_trial.params

    # Make the experiment config dictionary
    config = {}

    config['device'] = device
    config['dataset_path'] = dataset_path

    config['model_type'] = model_type
    config['alarm_type'] = alarm_type
    config['metric'] = metric
    config['required_warning_time'] = int(required_warning_time_ms)/1000

    config['hyperparameters'] = hyperparameters

    # Remove the previous experiment if it exists
    model_file = f"models/{device}/{dataset_path}/{study_name}.pkl"
    try:
        os.remove(model_file)
    except FileNotFoundError:
        pass

    # Create the experiment
    experiment = Experiment(config, 'test')

    # Cache the basic metrics
    experiment.get_critical_metrics_vs_thresholds()
    experiment.get_critical_metrics_vs_false_alarm_rates()

    # Save experiment results to file
    experiment_name = f"{model_type}_{alarm_type}_{metric}_{int(required_warning_time*1000)}ms_experiment"

    # Make directory if it doesn't already exist
    directory_name = f"models/{device}/{dataset_path}/experiments"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    with open(f"{directory_name}/{experiment_name}.pkl", 'wb') as f:
        dill.dump(experiment, f)
