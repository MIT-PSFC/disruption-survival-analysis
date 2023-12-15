import os
import sys

import dill
import numpy as np

from disruption_survival_analysis.manage_datasets import print_memory_usage

def main(device, dataset_path, model_type, alarm_type, metric, required_warning_time_ms, bootstrap_iterations, working_directory=None):
    # If an optional seventh argument is provided, change the working directory to that
    if working_directory is not None:
        os.chdir(working_directory)

    print_memory_usage("Coalesce Before Loading Slices")

    slice_dir = f"results/{device}/{dataset_path}/bootstraps/{model_type}_{alarm_type}_{metric}_{required_warning_time_ms}ms"
    slice_name_base = f"slice_"
    slice_names = [slice_name_base + str(i) + ".pkl" for i in range(bootstrap_iterations)]

    # Where the answers are stored
    tars_list = []
    fars_list = []
    warns_list = []

    for slice_name in slice_names:
        path = f"{slice_dir}/{slice_name}"
        try:
            slice_data = dill.load(open(path, 'rb'))
            false_alarm_rates = slice_data['false_alarm_rates']
            true_alarm_metrics = slice_data['true_alarm_metrics']
            warning_time_metrics = slice_data['warning_time_metrics']

        except FileNotFoundError:
            sys.stdout.write(f"Could not find {path}!\n")
            return
        tars_list.append(true_alarm_metrics['avg'])
        fars_list.append(false_alarm_rates)
        warns_list.append(warning_time_metrics['iqm'])
    
    print_memory_usage("Bootstrap After Getting Results")

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

    # Compute the median, upper quartile, lower quartile, max, and min warning times at each unique false alarm rate
    median_warns = np.median(interp_warns_list, axis=0)
    upper_warns = np.percentile(interp_warns_list, 75, axis=0)
    lower_warns = np.percentile(interp_warns_list, 25, axis=0)

    # Save the bootstrapped metrics
    bootstrapped_metrics = {}
    bootstrapped_metrics['fars'] = unique_fars

    bootstrapped_metrics['mean_tars'] = mean_tars
    bootstrapped_metrics['upper_tars'] = upper_tars
    bootstrapped_metrics['lower_tars'] = lower_tars
    #bootstrapped_metrics['max_tars'] = max_tars
    #bootstrapped_metrics['min_tars'] = min_tars

    bootstrapped_metrics['median_warns'] = median_warns
    bootstrapped_metrics['upper_warns'] = upper_warns
    bootstrapped_metrics['lower_warns'] = lower_warns
    #bootstrapped_metrics['max_warns'] = max_warns
    #bootstrapped_metrics['min_warns'] = min_warns

    # Make directory if it doesn't already exist
    directory_name = f"results/{device}/{dataset_path}/bootstraps"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    bootstrap_name = f"{model_type}_{alarm_type}_{metric}_{required_warning_time_ms}ms_bootstrap"

    with open(f"{directory_name}/{bootstrap_name}.pkl", 'wb') as f:
        dill.dump(bootstrapped_metrics, f)

    sys.stdout.write("Saved bootstrapped metrics")
    

if __name__ == "__main__":
    # Get the device, dataset path, model, alarm, metric, min_warning_time, and working directory from the command line
    device = sys.argv[1]
    dataset_path = sys.argv[2]
    model_type = sys.argv[3]
    alarm_type = sys.argv[4]
    metric = sys.argv[5]
    required_warning_time_ms = sys.argv[6]
    allocated_cpus = int(sys.argv[7])
    try:
        working_directory = sys.argv[8]
    except:
        working_directory = None

    main(device, dataset_path, model_type, alarm_type, metric, required_warning_time_ms, allocated_cpus, working_directory)