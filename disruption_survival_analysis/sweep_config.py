import os
import yaml

from disruption_survival_analysis.Experiments import Experiment
from disruption_survival_analysis.experiment_utils import load_experiment_config

# Dictionary for all the various hyperparameters which can be swept over
hyperparameter_ranges = {
    "batch_size": { # Default: 128
        "min": 64,
        "max": 256,
        "distribution": "int"
    },
    "class_time": { # Time in seconds before a disruption labeled as 'disruptive
        "min": 0.001,
        "max": 0.4,
        "distribution": "float",
        "log": False
    },
    "criterion": { # Function to measure quality of a split
        "values": ["gini", "entropy", "log_loss"],
        "distribution": "categorical"
    },
    "epochs": { # Default: 50
        "min": 50,
        "max": 200,
        "distribution": "int"
    },
    "distribution": {
        "values": ["Weibull", "LogNormal"],
        "distribution": "categorical"
    },
    "fit_time": { # Time in seconds to look in the past for linearly extrapolating the risk into the future
        # In Alex's paper used 0.05, 0.1, 0.2
        "min": 0.001,
        "max": 0.3,
        "distribution": "float",
        "log": False
    },
    "k": {
        "min": 1,
        "max": 6,
        "distribution": "int"
    },
    "l2": { # Default: 0.0001
        "min": 0.00001,
        "max": 0.01,
        "distribution": "float",
        "log": True
    },
    "layer_depth": {
        "min": 1,
        "max": 5,
        "distribution": "int"
    },
    "layer_width": {
        "min": 50,
        "max": 1000,
        "distribution": "int"
    },
    "learning_rate": { # Default: 0.001
        "min": 0.00001,
        "max": 0.01,
        "distribution": "float",
        "log": True
    },
    "max_features": {
        "values": ["sqrt", "log2"],
        "distribution": "categorical"
    },
    "min_samples_leaf": { # Minimum number of samples for each node ceil(value * n_samples)
        "min": 0.0001,
        "max":0.9,
        "distribution": "float",
        "log": False
    },
    "min_samples_split": { # Minimum number of samples for each split ceil(value * n_samples)
        "min": 0.0001,
        "max": 0.9,
        "distribution": "float",
        "log": False
    },
    "n_estimators": { # Number of trees in the forest. Default: 100
        "min": 50,
        "max": 1000,
        "distribution": "int"
    },
    "horizon": { # How many seconds into the future to predict
        "min": 0.001,
        "max": 0.3,
        "distribution": "float",
        "log": False
    },
    "temperature": {
        "min": 0.5,
        "max": 1.5,
        "distribution": "float",
        "log": False
    }
}

# Dictionary for which hyperparameters to sweep over for each model
model_hyperparameters = {
    "cph": ["l2",
            "horizon"],
    "dcph": ["batch_size",
            "epochs",
            "horizon",
            "layer_depth",
            "layer_width",
            "learning_rate"],
    "dsm": ["batch_size", 
            "distribution", 
            "epochs",
            "horizon",
            "k", 
            "layer_depth", 
            "layer_width",
            "learning_rate", 
            "max_features",
            "temperature"],
    "rf": ["class_time",
           "criterion",
           "max_features",
           "min_samples_leaf",
           "min_samples_split",
           "n_estimators"],
    "km": ["class_time",
           "criterion",
           "fit_time",
           "horizon",
           "max_features",
           "min_samples_leaf",
           "min_samples_split",
           "n_estimators"],
}

def make_sweep_config(device, dataset_path, model_type, alarm_type, metric, required_warning_time):
    # Make a dictionary for the sweep configuration
    sweep_config = {}

    sweep_config["device"] = device
    sweep_config["dataset_path"] = dataset_path

    sweep_config["model_type"] = model_type
    sweep_config["alarm_type"] = alarm_type
    sweep_config["metric"] = metric
    sweep_config["required_warning_time"] = required_warning_time
    
    hyperparameters = {}
    for hyperparameter in model_hyperparameters[model_type]:
        hyperparameters[hyperparameter] = hyperparameter_ranges[hyperparameter]
    sweep_config["hyperparameters"] = hyperparameters

    return sweep_config

def write_sweep_config(sweep_config):
    # Write the sweep config to a file
    device = sweep_config["device"]
    dataset_path = sweep_config["dataset_path"]
    model_type = sweep_config["model_type"]
    alarm_type = sweep_config["alarm_type"]
    metric = sweep_config["metric"]
    required_warning_time = sweep_config["required_warning_time"]

    sweep_config_name = f"{model_type}_{alarm_type}_{metric}_{int(required_warning_time*1000)}ms_sweep"

    directory_name = f"models/{device}/{dataset_path}/sweeps"

    # Make directory if it doesn't already exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    with open(f"{directory_name}/{sweep_config_name}.yaml", "w") as f:
        yaml.dump(sweep_config, f)


def create_experiment_groups(devices, dataset_paths, models, alarms, metrics, min_warning_times):
    # Create groups of experiments
    # This is used for setting up dictionaries of experiments to plot
    # For example, compare all the experiments using the same minimum warning time against eachother
    experiment_groups = {}
    for device in devices:
        for dataset_path in dataset_paths:
            for model in models:
                for alarm in alarms:
                    for metric in metrics:
                        for min_warning_time in min_warning_times:
                            # Load config for experiment 
                            config = load_experiment_config(device, dataset_path, model, alarm, metric, min_warning_time)
                            # Create test experiment from config
                            experiment = Experiment(config, 'test')

                            try:
                                if experiment_groups[device] is None:
                                    experiment_groups[device] = []
                            except KeyError:
                                experiment_groups[device] = []
                            experiment_groups[device].append(experiment)

                            try:
                                if experiment_groups[dataset_path] is None:
                                    experiment_groups[dataset_path] = []
                            except KeyError:
                                experiment_groups[dataset_path] = []
                            experiment_groups[dataset_path].append(experiment)
                            
                            try:
                                if experiment_groups[model] is None:
                                    experiment_groups[model] = []
                            except KeyError:
                                experiment_groups[model] = []
                            experiment_groups[model].append(experiment)

                            try:
                                if experiment_groups[alarm] is None:
                                    experiment_groups[alarm] = []
                            except KeyError:
                                experiment_groups[alarm] = []
                            experiment_groups[alarm].append(experiment)

                            try:
                                if experiment_groups[metric] is None:
                                    experiment_groups[metric] = []
                            except KeyError:
                                experiment_groups[metric] = []
                            experiment_groups[metric].append(experiment)

                            try:
                                if experiment_groups[min_warning_time] is None:
                                    experiment_groups[min_warning_time] = []
                            except KeyError:
                                experiment_groups[min_warning_time] = []
                            experiment_groups[min_warning_time].append(experiment)

    return experiment_groups

def get_experiments(experiment_groups, keys1, keys2=None, keys3=None, keys4=None):
    """Get list of experiments that match all keys"""
    experiment_list = []
    for experiment in experiment_groups[keys1[0]]:
        if all([experiment in experiment_groups[key] for key in keys1]):
            experiment_list.append(experiment)
    if keys2 is not None:
        for experiment in experiment_groups[keys2[0]]:
            if all([experiment in experiment_groups[key] for key in keys2]):
                experiment_list.append(experiment)
    if keys3 is not None:
        for experiment in experiment_groups[keys3[0]]:
            if all([experiment in experiment_groups[key] for key in keys3]):
                experiment_list.append(experiment)
    if keys4 is not None:
        for experiment in experiment_groups[keys4[0]]:
            if all([experiment in experiment_groups[key] for key in keys4]):
                experiment_list.append(experiment)
    
    return experiment_list