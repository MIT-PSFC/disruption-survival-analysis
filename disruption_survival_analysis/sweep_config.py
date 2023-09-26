import os
import yaml

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
        "max": 200,
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
        "max": 200,
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

    # Make directory if it doesn't already exist
    if not os.path.exists(f"models/{device}/{dataset_path}"):
        os.makedirs(f"models/{device}/{dataset_path}")

    with open(f"models/{device}/{dataset_path}/{sweep_config_name}.yaml", "w") as f:
        yaml.dump(sweep_config, f)