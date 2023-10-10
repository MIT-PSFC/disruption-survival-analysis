# Exactly the same code as corresponding notebook, but for whatever reason engaging does not like notebooks

from disruption_survival_analysis.sweep_config import make_sweep_config, write_sweep_config

# Datasets to use
devices = ["cmod"]
dataset_paths = ["sql_all_no_ufo", "sql_all_no_ufo/stack_5", "sql_all_no_ufo/stack_10"]
#devices = ["synthetic"]
#dataset_paths = ["test"]

# List of models to create sweeps for
# cph, dcph, dcm, dsm, rf, km
model_types = ["cph", "dsm", "rf", "km"]

# List of alarm types to use
# sthr, hyst, ettd, ethy
alarm_types = ["sthr"]

# List of validation metrics to use
# auroc, auwtc, maxf1
metrics = ["auroc", "aumal"]

# List of required warning times to train on (in seconds)
required_warning_times = [0.01, 0.05, 0.1]

for device in devices:
    for dataset_path in dataset_paths:
        for model_type in model_types:
            for alarm_type in alarm_types:
                for metric in metrics:
                    for required_warning_time in required_warning_times:
                        sweep_config = make_sweep_config(device, dataset_path, model_type, alarm_type, metric, required_warning_time)
                        write_sweep_config(sweep_config)