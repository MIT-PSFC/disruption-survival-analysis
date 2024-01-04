import os
import sys

import dill
import numpy as np

from disruption_survival_analysis.Experiments import Experiment
from disruption_survival_analysis.manage_datasets import print_memory_usage
from disruption_survival_analysis.experiment_utils import load_experiment_config
from disruption_survival_analysis.critical_metrics import interquartile_mean

NUM_BOOTSTRAPS = 50

def main(device, dataset_path, model_type, alarm_type, metric, required_warning_time_ms, working_directory=None):
    # If an optional sixth argument is provided, change the working directory to that
    if working_directory is not None:
        os.chdir(working_directory)

    required_warning_time = int(required_warning_time_ms)/1000
    config = load_experiment_config(device, dataset_path, model_type, alarm_type, metric, required_warning_time)

    print_memory_usage("Simple RMST Before Creating Experiment")

    # Create the experiment
    experiment = Experiment(config, 'test')
    sys.stdout.write("Created experiment\n")

    print_memory_usage("Simple RMST After Creating Experiment")

    disruptive_rmst_diffs, non_disruptive_rmst_diffs = experiment.get_simple_rmst_integrals()
    all_diffs = np.concatenate((disruptive_rmst_diffs, non_disruptive_rmst_diffs))

    results = {}
    results['disruptive_rmst_diffs'] = disruptive_rmst_diffs
    results['non_disruptive_rmst_diffs'] = non_disruptive_rmst_diffs

    print_memory_usage("Simple RMST After Getting Results")

    # Bootstrap the RMST integrals individually and together

    sampled_disruptive_array = np.zeros(NUM_BOOTSTRAPS)
    sampled_non_disruptive_array = np.zeros(NUM_BOOTSTRAPS)
    sampled_all_array = np.zeros(NUM_BOOTSTRAPS)

    np.random.seed(0)
    for i in range(NUM_BOOTSTRAPS):
        sampled_disruptive = np.random.choice(disruptive_rmst_diffs, len(disruptive_rmst_diffs))
        sampled_non_disruptive = np.random.choice(non_disruptive_rmst_diffs, len(non_disruptive_rmst_diffs))
        sampled_all = np.random.choice(all_diffs, len(all_diffs))

        # Calculate the mean of the sampled arrays
        sampled_disruptive_array[i] = np.mean(sampled_disruptive)
        sampled_non_disruptive_array[i] = np.mean(sampled_non_disruptive)
        sampled_all_array[i] = np.mean(sampled_all)

    # Report the mean, standard deviation, median, upper, and lower quartiles, and interquartile mean of the sampled arrays

    disruptive_results = {}
    disruptive_results['avg'] = np.mean(sampled_disruptive_array)
    disruptive_results['std'] = np.std(sampled_disruptive_array)
    disruptive_results['med'] = np.median(sampled_disruptive_array)
    disruptive_results['iq1'] = np.percentile(sampled_disruptive_array, 25)
    disruptive_results['iq3'] = np.percentile(sampled_disruptive_array, 75)
    disruptive_results['iqm'] = interquartile_mean(sampled_disruptive_array)
    results['disruptive_results'] = disruptive_results

    non_disruptive_results = {}
    non_disruptive_results['avg'] = np.mean(sampled_non_disruptive_array)
    non_disruptive_results['std'] = np.std(sampled_non_disruptive_array)
    non_disruptive_results['med'] = np.median(sampled_non_disruptive_array)
    non_disruptive_results['iq1'] = np.percentile(sampled_non_disruptive_array, 25)
    non_disruptive_results['iq3'] = np.percentile(sampled_non_disruptive_array, 75)
    non_disruptive_results['iqm'] = interquartile_mean(sampled_non_disruptive_array)
    results['non_disruptive_results'] = non_disruptive_results

    all_results = {}
    all_results['avg'] = np.mean(sampled_all_array)
    all_results['std'] = np.std(sampled_all_array)
    all_results['med'] = np.median(sampled_all_array)
    all_results['iq1'] = np.percentile(sampled_all_array, 25)
    all_results['iq3'] = np.percentile(sampled_all_array, 75)
    all_results['iqm'] = interquartile_mean(sampled_all_array)
    results['all_results'] = all_results

    print_memory_usage("Simple RMST after bootstrapping")

    rmst_dir = f"{model_type}_{alarm_type}_{metric}_{required_warning_time_ms}ms"

    # Make directory if it doesn't already exist
    directory_name = f"results/{device}/{dataset_path}/simple_rmst/{rmst_dir}"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    with open(f"{directory_name}/all_rmst_results.pkl", 'wb') as f:
        dill.dump(results, f)

    sys.stdout.write("Saved all RMST integrals")

if __name__ == "__main__":
    # Get the device, dataset path, model, alarm, metric, min_warning_time, and working directory from the command line
    device = sys.argv[1]
    dataset_path = sys.argv[2]
    model_type = sys.argv[3]
    alarm_type = sys.argv[4]
    metric = sys.argv[5]
    required_warning_time_ms = sys.argv[6]
    try:
        working_directory = sys.argv[7]
    except:
        working_directory = None

    main(device, dataset_path, model_type, alarm_type, metric, required_warning_time_ms, working_directory)

