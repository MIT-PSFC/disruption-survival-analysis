import os
import sys

import dill
import numpy as np
import datetime

BOOTSTRAP_ITERATIONS = 50

if __name__ == "__main__":
    # Get the device, dataset path, model, alarm, metric, and min_warning_time from the command line
    device = sys.argv[1]
    dataset_path = sys.argv[2]
    model_type = sys.argv[3]
    alarm_type = sys.argv[4]
    metric = sys.argv[5]
    min_warning_time_ms = sys.argv[6]
    
    experiment_name = f"{model_type}_{alarm_type}_{metric}_{min_warning_time_ms}ms_experiment"

    # Load the experiment
    experiment_path = f"models/{device}/{dataset_path}/experiments/{experiment_name}.pkl"

    with open(experiment_path, 'rb') as f:
        experiment = dill.load(f)

    print(f"Loaded experiment {experiment_name}")

    # Calculate the bootstrapped metrics
    tars_list = []
    fars_list = []
    warns_list = []

    for i in range(BOOTSTRAP_ITERATIONS):
        false_alarm_rates, true_alarm_rates, avg_warning_times, _ = experiment.get_critical_metrics_vs_false_alarm_rates(bootstrap_seed=i)
        tars_list.append(true_alarm_rates)
        fars_list.append(false_alarm_rates)
        warns_list.append(avg_warning_times)
        # Print when each bootstrap iteration is finished
        print(f"Finished bootstrap iteration {i} at {datetime.datetime.now()}")

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
    directory_name = f"models/{device}/{dataset_path}/bootstrapped_metrics"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    bootstrap_name = f"{model_type}_{alarm_type}_{metric}_{min_warning_time_ms}ms_bootstrap"

    with open(f"{directory_name}/{bootstrap_name}.pkl", 'wb') as f:
        dill.dump(bootstrapped_metrics, f)

    print("Saved bootstrapped metrics")