# Dictionary for all the various hyperparameters which can be swept over
hyperparameter_ranges = {
    "l2": { # Default: 0.0001
        "min": 0.00001,
        "max": 0.01,
        "distribution": "log_uniform_values"
    },
    "batch_size": { # Default: 128
        "min": 64,
        "max": 256,
    },
    "epochs": { # Default: 50
        "min": 50,
        "max": 200,
    },
    "learning_rate": { # Default: 0.001
        "min": 0.0001,
        "max": 0.01,
        "distribution": "log_uniform_values"
    },
    "horizon": {
        "min": 0.001,
        "max": 0.5,
    }
}

# Dictionary for which hyperparameters to sweep over for each model
model_hyperparameters = {
    "cph": ["l2", "horizon"],
    "dsm": ["batch_size", "epochs", "learning_rate", "horizon"],
}


# Base project name
project_name = "2023-08-11 Big Refactor"
# Datasets to use
devices = ["cmod"]
dataset_paths = ["no_ufo_flattop_7736_shots_6%_disruptive", 
                 "no_ufo_flattop_7736_shots_6%_disruptive/stack_4"]

# List of models to include in this sweep
# cph, dcph, dcm, dsm, rf, km
models = ["cph", "dsm"]



# List of alarm types to use
# sthr, hyst, ettd, ethy

# List of validation metrics to use
# auroc, auwtc, maxf1, etint

# List of minimum warning times to train on
min_warning_times = [0.1, 0.02]