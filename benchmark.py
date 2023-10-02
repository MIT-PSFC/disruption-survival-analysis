from disruption_survival_analysis.sweep_config import create_experiment_groups, get_experiments, load_experiment_config

from disruption_survival_analysis.plot_experiments import plot_roc_curve

from disruption_survival_analysis.Experiments import Experiment

import time

#device = 'synthetic'
#dataset_path = 'test_2'
device = 'cmod'
dataset_path = 'preliminary_dataset_no_ufo'

# models, alarms, metrics, and minimum warning times to use
models = ['dsm', 'rf']
alarms = ['athr']
metrics = ['auroc']
min_warning_times = [0.02, 0.1]

# Load models and create experiments
#experiment_groups = create_experiment_groups(device, dataset_path, models, alarms, metrics, min_warning_times)

#required_warning_time = 0.02
#experiment_list = get_experiments(experiment_groups, ['dsm', 'athr', 'auroc', required_warning_time], ['rf', 'athr', 'auroc', required_warning_time])

#plot_roc_curve(experiment_list, required_warning_time=required_warning_time, debug=True)

experiment_config = load_experiment_config(device, dataset_path, 'rf', 'athr', 'auroc', 0.02)
experiment = Experiment(experiment_config, 'test')

start_time = time.time()
#direct_true_alarm_rates, direct_false_alarm_rates, direct_avg_warning_times, direct_std_warning_times = experiment.get_critical_metrics_vs_thresholds(horizon=0.05, required_warning_time=0.02)

_, _, _, _ = experiment.get_critical_metrics_vs_false_alarm_rate(horizon=0.05, required_warning_time=0.02)

print('Direct computation took {} seconds'.format(time.time() - start_time))