import yaml

# Dictionary for all the various hyperparameters which can be swept over
hyperparameter_ranges = {
    "batch_size": { # Default: 128
        "min": 64,
        "max": 256,
    },
    "class_time": { # Time in seconds before a disruption labeled as 'disruptive
        "min": 0.001,
        "max": 0.5,
    },
    "criterion": { # Function to measure quality of a split
        "values": ["gini", "entropy", "log_loss"],
        "distribution": "categorical"
    },
    "epochs": { # Default: 50
        "min": 50,
        "max": 200,
    },
    "distribution": {
        "values": ["Weibull", "LogNormal"],
        "distribution": "categorical"
    },
    "k": {
        "min": 1,
        "max": 6,
    },
    "l2": { # Default: 0.0001
        "min": 0.00001,
        "max": 0.01,
        "distribution": "log_uniform_values"
    },
    "layer_depth": {
        "min": 1,
        "max": 5,
    },
    "layer_width": {
        "min": 50,
        "max": 200,
    },
    "learning_rate": { # Default: 0.001
        "min": 0.00001,
        "max": 0.01,
        "distribution": "log_uniform_values"
    },
    "max_features": {
        "values": ["sqrt", "log2"],
        "distribution": "categorical"
    },
    "min_samples_leaf": { # Minimum number of samples for each node ceil(value * n_samples)
        "min": 0.0001,
        "max":0.9
    },
    "min_samples_split": { # Minimum number of samples for each split ceil(value * n_samples)
        "min": 0.0001,
        "max": 0.9
    },
    "n_estimators": { # Number of trees in the forest. Default: 100
        "min": 50,
        "max": 200
    },
    "horizon": { # How many seconds into the future to predict
        "min": 0.001,
        "max": 0.5,
    },
    "temperature": {
        "min": 0.5,
        "max": 1.5,
    }
}

# Dictionary for which hyperparameters to sweep over for each model
model_hyperparameters = {
    "cph": ["l2", 
            "horizon"],
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
           "n_estimators"]
}

# Base project name
project_name = "2023-08-16 Tests"
# Datasets to use
devices = ["synthetic"]
dataset_paths = ["synthetic100"]

# List of models to include in this sweep
# cph, dcph, dcm, dsm, rf, km
models = ["dsm", "rf"]

# List of alarm types to use
# sthr, hyst, ettd, ethy
alarm_types = ["sthr", "hyst"]

# List of validation metrics to use
# auroc, auwtc, maxf1, etint
metrics = ["auroc"]

# List of required warning times to train on (in seconds)
required_warning_times = [0.1, 0.02]

def make_sweep_config(device, dataset_path, model, alarm_type, metric, required_warning_time):
    sweep_name = f"{model}-{alarm_type}-{metric}-{int(required_warning_time*1000)}ms-{device}-{dataset_path}"

    sweep_config = {}

    # Write the frontmatter
    sweep_config["command"] = ["${env}", "python", "${program}", "${args}"]
    sweep_config["program"] = "wandb_job.py"
    sweep_config["method"] = "bayes"
    sweep_config["name"] = sweep_name
    sweep_config["description"] = project_name

    # Determine which direction the metric should go
    metric_config = {"name": metric}
    if metric in ["auroc", "auwtc", "maxf1"]:
        metric_config["goal"] = "maximize"
    else:
        metric_config["goal"] = "minimize"

    sweep_config["metric"] = metric_config
    
    # Write the parameter config
    parameters = {}

    parameters["aa-device"] = {"value": device,
                                "distribution": "constant"}
    parameters["aa-dataset-path"] = {"value": dataset_path,
                                    "distribution": "constant"}
    parameters["aa-model-type"] = {"value": model,
                                    "distribution": "constant"}
    parameters["ab-alarm-type"] = {"value": alarm_type,
                                    "distribution": "constant"}
    parameters["ab-metric"] = {"value": metric,
                                "distribution": "constant"}
    parameters["ab-required-warning-time"] = {"value": required_warning_time,
                                                "distribution": "constant"}
    
    # Add the hyperparameters to the config
    for hyperparameter in model_hyperparameters[model]:
        parameters[hyperparameter] = hyperparameter_ranges[hyperparameter]

    sweep_config["parameters"] = parameters

    return sweep_config

def write_sweep_config(sweep_config):
    # Write the sweep config to a file
    device = sweep_config["parameters"]["aa-device"]["value"]
    dataset_path = sweep_config["parameters"]["aa-dataset-path"]["value"]
    model = sweep_config["parameters"]["aa-model-type"]["value"]
    alarm_type = sweep_config["parameters"]["ab-alarm-type"]["value"]
    metric = sweep_config["parameters"]["ab-metric"]["value"]
    required_warning_time = sweep_config["parameters"]["ab-required-warning-time"]["value"]

    sweep_config_name = f"{model}_{alarm_type}_{metric}_{int(required_warning_time*1000)}ms_sweep"

    with open(f"models/{device}/{dataset_path}/{sweep_config_name}.yaml", "w") as f:
        yaml.dump(sweep_config, f)

for device in devices:
    for dataset_path in dataset_paths:
        for model in models:
            for alarm_type in alarm_types:
                for metric in metrics:
                    for required_warning_time in required_warning_times:
                        sweep_config = make_sweep_config(device, dataset_path, model, alarm_type, metric, required_warning_time)
                        write_sweep_config(sweep_config)