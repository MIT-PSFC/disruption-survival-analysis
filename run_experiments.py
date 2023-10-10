from disruption_survival_analysis.sweep_config import create_experiment_groups

#device = 'synthetic'
#dataset_path = 'test'
device = 'cmod'
dataset_path = 'preliminary_dataset_no_ufo/stack_6'

# models, alarms, metrics, and minimum warning times to use
models = ['cph', 'dsm', 'rf', 'km']
#models = ['rf']
alarms = ['sthr']
metrics = ['auroc', 'auwtc']
min_warning_times = [0.01, 0.05, 0.1, 0.2]

# Load models and create experiments
experiment_groups = create_experiment_groups(device, dataset_path, models, alarms, metrics, min_warning_times)