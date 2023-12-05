import os
import sys

import dill
import numpy as np
import datetime
import time

from multiprocessing import Pool

from disruption_survival_analysis.manage_datasets import print_memory_usage

BOOTSTRAP_ITERATIONS = 50
ALLOCATED_CPUS = 20


if __name__ == "__main__":
    # Get the device, dataset path, model, alarm, metric, and min_warning_time from the command line
    device = sys.argv[1]
    dataset_path = sys.argv[2]
    model_type = sys.argv[3]
    alarm_type = sys.argv[4]
    metric = sys.argv[5]
    min_warning_time_ms = sys.argv[6]

    # If an optional seventh argument is provided, change the working directory to that
    try:
        os.chdir(sys.argv[7])
    except:
        pass
    
    experiment_name = f"{model_type}_{alarm_type}_{metric}_{min_warning_time_ms}ms_experiment"

    # Load the experiment
    experiment_path = f"models/{device}/{dataset_path}/experiments/{experiment_name}.pkl"

    try:
        with open(experiment_path, 'rb') as f:
            experiment = dill.load(f)
    except:
        sys.stdout.write("Unable to load experiment due to pickle file not being read. Waiting 5 min")
        time.sleep(5*60)

        with open(experiment_path, 'rb') as f:
            experiment = dill.load(f)

    sys.stdout.write(f"Loaded experiment {experiment_name}")

    # Where the answers are stored
    tars_list = []
    fars_list = []
    warns_list = []

    pool = Pool(ALLOCATED_CPUS)
    sys.stdout.write(f"Created POOL with {ALLOCATED_CPUS} processes\n")

    # Array where all threads put their results asynchronously
    results = []

    for i in range(BOOTSTRAP_ITERATIONS):
        results.append(pool.apply_async(experiment.get_critical_metrics_vs_false_alarm_rates, [None, None, i]))
        print_memory_usage()

    for result in results:
        false_alarm_rates, true_alarm_rates, avg_warning_times, _ = result.get(timeout=100000)
        tars_list.append(true_alarm_rates)
        fars_list.append(false_alarm_rates)
        warns_list.append(avg_warning_times)
        print_memory_usage()

    # Find all the unique false alarm rates and sort them
    unique_fars = np.unique(np.concatenate(fars_list))
    unique_fars.sort()

    # Interpolate the true alarm rates for each bootstrap at the unique false alarm rates
    interp_tars_list = []
    interp_warns_list = []
    for i, tars in enumerate(tars_list):
        interp_tars_list.append(np.interp(unique_fars, fars_list[i], tars_list[i]))
        interp_warns_list.append(np.interp(unique_fars, fars_list[i], warns_list[i]))

    # Compute the mean, upper quartile, lower quartile, max, and min true alarm rates at each unique false alarm rate
    mean_tars = np.mean(interp_tars_list, axis=0)
    upper_tars = np.percentile(interp_tars_list, 75, axis=0)
    lower_tars = np.percentile(interp_tars_list, 25, axis=0)
    max_tars = np.max(interp_tars_list, axis=0)
    min_tars = np.min(interp_tars_list, axis=0)

    # Compute the mean, upper quartile, lower quartile, max, and min warning times at each unique false alarm rate
    mean_warns = np.mean(interp_warns_list, axis=0)
    upper_warns = np.percentile(interp_warns_list, 75, axis=0)
    lower_warns = np.percentile(interp_warns_list, 25, axis=0)
    max_warns = np.max(interp_warns_list, axis=0)
    min_warns = np.min(interp_warns_list, axis=0)

    # Save the bootstrapped metrics
    bootstrapped_metrics = {}
    bootstrapped_metrics['fars'] = unique_fars

    bootstrapped_metrics['mean_tars'] = mean_tars
    bootstrapped_metrics['upper_tars'] = upper_tars
    bootstrapped_metrics['lower_tars'] = lower_tars
    bootstrapped_metrics['max_tars'] = max_tars
    bootstrapped_metrics['min_tars'] = min_tars

    bootstrapped_metrics['mean_warns'] = mean_warns
    bootstrapped_metrics['upper_warns'] = upper_warns
    bootstrapped_metrics['lower_warns'] = lower_warns
    bootstrapped_metrics['max_warns'] = max_warns
    bootstrapped_metrics['min_warns'] = min_warns

    # Make directory if it doesn't already exist
    directory_name = f"models/{device}/{dataset_path}/bootstraps"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    bootstrap_name = f"{model_type}_{alarm_type}_{metric}_{min_warning_time_ms}ms_bootstrap"

    with open(f"{directory_name}/{bootstrap_name}.pkl", 'wb') as f:
        dill.dump(bootstrapped_metrics, f)

    sys.stdout.write("Saved bootstrapped metrics")