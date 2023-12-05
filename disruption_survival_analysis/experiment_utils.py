# Functions used by experiments but not actually part of the experiments

import os
import numpy as np
import optuna
import yaml

from sklearn.ensemble import RandomForestClassifier
from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
from disruption_survival_analysis.manage_datasets import load_features_outcomes

# Labeling data

def label_shot_data(shot_data, disrupt, disruptive_window):
    """
    Label the data as disruptive or not disruptive based on the disruptive window
    Parameters
    ----------
    data : pandas.DataFrame
        The data to label
    disrupt : bool
        If the shot is disruptive
    disruptive_window : float
        Time before a disruption to label shot data as disruptive (in seconds)
    Returns
    -------
    labeled_data : numpy.ndarray
        An array of booleans indicating if the time slice is disruptive or not
    """

    if disrupt:
        # If the shot disrupts, label all time slices up to
        # disruptive_window seconds before the disruption as non-disruptive
        # and all time slices after disruptive_window seconds before the disruption as disruptive
        # Labels are either 0 (non-disruptive) or 1 (disruptive)
        disruption_time = shot_data['time'].max()
        labeled_data = np.array(shot_data['time'] > (disruption_time - disruptive_window)).astype(int)
    else:
        # If the shot is not disruptive, label all time slices as non-disruptive
        labeled_data = np.zeros(len(shot_data))

    return labeled_data

def timeslice_micro_avg(device, dataset_path, model, experiment_type):
    """
    Calculates the micro-averaged metric for a given model on a given dataset
    If the model is a survival model, calculates the IBS metric
    If the model is a classifier, calculates the mean accuracy

    Parameters
    ----------
    device : str
        The device used for training and evaluation
    dataset_path : str
        The path to the dataset to use for evaluation
    model : SurvivalModel or RandomForestClassifier
        The model to evaluate
    experiment_type : str
        The type of experiment is being run, either 'val' or 'test'

    Returns
    -------
    metric_val : float
        The micro-averaged metric for the model on the dataset
    """
    # Load either validation or test data
    x_set, y_set = load_features_outcomes(device, dataset_path, experiment_type)

    # Validation times are hardcoded for now
    val_times = [0.1, 0.02]

    # Evaluate the model on the validation set
    try:
        if isinstance(model, SurvivalModel):
            predictions_val = model.predict_survival(x_set, val_times)
            _, y_train = load_features_outcomes(device, dataset_path, 'train')
            if np.isnan(predictions_val).any():
                return None
            else:
                metric_val = survival_regression_metric('ibs', y_set, predictions_val, val_times, y_train)
        elif isinstance(model, RandomForestClassifier):
            metric_val = model.score(x_set, y_set)
        else:
            return None
    except:
        metric_val = None

    return metric_val

def area_under_curve(x_vals, y_vals, x_cutoff=None):
    """
    Calculates the area under the curve for a given set of x and y values
    If x_cutoff is specified, the area under the curve is calculated only up to x_cutoff

    Parameters
    ----------
    x_vals : array-like
        The x values for the curve
    y_vals : array-like
        The y values for the curve
    x_cutoff : float
        The x value to cutoff the curve at (will be evaluated as x < x_cutoff)
    
    Returns
    -------
    auc : float
        The area under the curve
    """

    # Limit the false alarm rate to be less than the x cutoff
    if x_cutoff is not None:
        # First, find the index of the values that are directly above and below the cutoff
        # This is done by finding the index of the first value that is above the cutoff
        # and then subtracting 1

        upper_index = None
        # Find the index of the first value that is above the cutoff
        for i, x_val in enumerate(x_vals):
            if x_val > x_cutoff:
                upper_index = i
                break
        # If the index is 0, then all the values are above the cutoff
        # In this case, return 0
        if upper_index is None or upper_index == 0:
            return 0
        # Otherwise, subtract 1 to get the index of the value directly below the cutoff
        else:
            lower_index = upper_index - 1

        # Hold the y value directly below the cutoff constant
        # This makes it so our metric is only influenced by values below the cutoff
        # In addition, if the average warning time is really good early on (i.e. the curve is very steep)
        # and there are no other false alarm rates until later, this will lead to a large area under the curve
        # which is exactly what we want
        y_cutoff = y_vals[lower_index]

        # Add the cutoff values to the x and y values
        x_vals = np.append(x_vals, x_cutoff)
        y_vals = np.append(y_vals, y_cutoff)

        y_vals = y_vals[x_vals <= x_cutoff]
        x_vals = x_vals[x_vals <= x_cutoff]
        

    # Sort the values by x
    sorted_indices = np.argsort(x_vals)
    x_vals = x_vals[sorted_indices]
    y_vals = y_vals[sorted_indices]

    # Calculate the area under the curve
    auc = np.trapz(y_vals, x_vals)

    return auc

def calculate_f1_scores(true_alarm_count_array, false_alarm_count_array, num_disruptive_shots):
    """
    Calculates the F1 scores for a given set of true alarm counts and false alarm counts

    Parameters
    ----------
    true_alarm_count_array : array-like
        The number of true alarms for each threshold
    false_alarm_count_array : array-like
        The number of false alarms for each threshold
    num_disruptive_shots : int
        The number of shots that disrupted

    Returns
    -------
    f1_scores : array-like
        The F1 score for each threshold
    
    """
    f1_scores = []
    for true_alarm_count, false_alarm_count in zip(true_alarm_count_array, false_alarm_count_array):
        missed_alarm_count = num_disruptive_shots - true_alarm_count
        f1_score = true_alarm_count/(true_alarm_count + 0.5*(missed_alarm_count + false_alarm_count))
        f1_scores.append(f1_score)

    return f1_scores

# Other functions

def load_experiment_config(device, dataset, model_type, alarm_type, metric, required_warning_time):
    """
    Load an experiment config dictionary.
    Either from a yaml file (first attempt) or from a study database (second attempt)
    Expects file to be one directory up from the current directory, in the 'models' folder

    Parameters
    ----------
    device : str
        The device to load the config for
    dataset : str
        The dataset to load the config for
    model_type : str
        The model type to load the config for.
        Choices are 'cph', 'dcph', 'dcm', 'dsm', 'rf', 'km'
    alarm_type : str
        The alarm type to load the config for.
        Choices are 'sthr'
    metric : str
        The metric to load the config for.
        Choices are 'tslic', 'auroc', 'auwtc'

    Returns
    -------
    experiment_config : dict
        The experiment config dictionary
    
    """
    print("---")
    print("Attempting to load hyperparameters from yaml file...")
    yaml_file_name = f"{model_type}_{alarm_type}_{metric}_{int(required_warning_time*1000)}ms.yaml"
    try:
        with open(f"models/{device}/{dataset}/configs/{yaml_file_name}", "r") as f:
            hyperparameters = yaml.load(f, Loader=yaml.FullLoader)['hyperparameters']
        print(f"Loaded hyperparameters for {device}/{dataset}/configs/{yaml_file_name}")
        print("---")
    except:
        print("YAML not found. Attempting to load hyperparameters from study database...")
        db_file_name = f"{model_type}_{alarm_type}_{metric}_{int(required_warning_time*1000)}ms_study.db"
        try:
            # Get the path to the database file
            full_path = f"models/{device}/{dataset}/studies/{db_file_name}"
            
            # Check if the database file exists
            with open(full_path, "r") as f:
                pass

            lock_obj = optuna.storages.JournalFileOpenLock(full_path)
            storage = optuna.storages.JournalStorage(
                optuna.storages.JournalFileStorage(full_path, lock_obj=lock_obj)
            )

            # Get the best trial from the study (expects there to be only one study in the database)
            study = optuna.load_study(study_name=None, storage=storage)

            hyperparameters = study.best_trial.params

            # Save the hyperparameters to a yaml file
            # Create the configs folder if it doesn't exist yet
            configs_folder = os.path.dirname(f"models/{device}/{dataset}/configs/{yaml_file_name}")
            if not os.path.exists(configs_folder):
                try:
                    os.makedirs(configs_folder)
                except:
                    pass

            with open(f"models/{device}/{dataset}/configs/{yaml_file_name}", "w") as f:
                yaml.dump({'hyperparameters': hyperparameters}, f)

            print(f"Loaded hyperparameters for {device}/{dataset}/studies/{db_file_name}")
            print(f"Best validation metric is {study.best_trial.value} from trial {study.best_trial.number}")
            print("---")
        except:
            print(f"Could not load hyperparameters for {device}/{dataset}/studies/{db_file_name}")
            print("---")
            hyperparameters = None

    # Make the experiment config dictionary
    config = {}

    config['device'] = device
    config['dataset_path'] = dataset

    config['model_type'] = model_type
    config['alarm_type'] = alarm_type
    config['metric'] = metric
    config['required_warning_time'] = required_warning_time

    config['hyperparameters'] = hyperparameters

    return config

