# Functions used by experiments but not actually part of the experiments

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
from manage_datasets import load_features_outcomes


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

def make_shot_lifetime_curve(times, disrupt, lifetime):
    """
    Create a shot lifetime curve from a list of times and a disruption time.
    The shot lifetime curve is a simplistic linear model of the shot lifetime,
    where non-disruptive shots have a constant lifetime throughout, 
    and disruptive shots have a lifetime that is constant until the end where
    it linearly decreases to 0

    Parameters
    ----------
    times : numpy.ndarray
        The times of each time slice in the shot
    disrupt : bool
        If the shot is disruptive
    disruptive_window : float
        Time before a disruption to label shot data as disruptive (in seconds)

    Returns
    -------
    shot_lifetime_curve: numpy.ndarray
        An array of what the anticipated shot lifetime should be with a simplistic linear model
    """

    if disrupt:
        # If the shot disrupts, the lifetime is constant until the end
        # where it linearly decreases to 0
        shot_lifetime_curve = np.ones(len(times)) * lifetime
        shot_lifetime_curve[times > (times.max() - lifetime)] = np.linspace(lifetime, 0, len(times[times > (times.max() - lifetime)]))
    else:
        # If the shot is not disruptive, all the timeslices get the same time
        shot_lifetime_curve = np.ones(len(times)) * lifetime

    return shot_lifetime_curve

# Calculating alarm times

def calculate_alarm_times(risk_at_time, thresholds):
    """
    Calculates the alarm times for a given shot with a simple threshold

    Parameters
    ----------
    risk_at_time : pandas.DataFrame
        The risk of disruption for each time slice in a single shot
        Should be sorted by time
        Should be transformed by the predictor's transformer
    thresholds : list of float
        The thresholds to use for determining if a disruption is imminent
        Expects a list of floats between 0 and 1, sorted from lowest to highest
        Disruption is predicted when the risk exceeds a threshold

    Returns
    -------
    alarm_times : list of float
        The times of alarm (predicted disruption)
        If no disruption is predicted, returns None in that position
    """
    
    # Make a copy of the thresholds to keep track of which have been used
    if isinstance(thresholds, np.ndarray):
        avail_thresholds = thresholds.tolist()
    else:
        avail_thresholds = thresholds.copy()

    alarm_times = []
    # Go through the shot data and find the first time the risk exceeds each threshold
    for i in range(len(risk_at_time)):
        # If there are no more thresholds, stop
        if len(avail_thresholds) == 0:
            break

        # If the risk ever exceeds the threshold, add the time to the list
        # and remove the threshold from the list
        # Then keep going until the risk is below the next threshold
        # or there are no more thresholds
        risk = risk_at_time.iloc[i]['risk']
        while risk > avail_thresholds[0]:
            alarm_times.append(risk_at_time.iloc[i]['time'])
            avail_thresholds.pop(0)
            if len(avail_thresholds) == 0:
                break
    
    # If there is a mismatch between alarm times and thresholds,
    # fill in the rest of the alarm times with None
    if len(alarm_times) < len(thresholds):
        for i in range(len(thresholds) - len(alarm_times)):
            alarm_times.append(None)

    # Return the alarm times
    return alarm_times

def calculate_alarm_times_hysteresis(risk_at_time, thresholds):
    """
    Calculates the alarm times for a given shot with hysterisis method
    If the 'disruptivity' output of the model goes above the upper threshold
    and remains above the lower threshold for the window length, a disruption
    is predicted

    Parameters
    ----------
    risk_at_time : pandas.DataFrame
        The risk of disruption for each time slice in a single shot
        Should be sorted by time
        Should be transformed by the predictor's transformer
    thresholds : list of tuple of (float, float, float)
        The thresholds to use for determining if a disruption is imminent
        Expects a list of tuples of the form (lower_threshold, upper_threshold, window_length)
        Disruption is predicted when the risk exceeds the upper threshold
        and remains above the lower threshold for the window length (Same implementation as ENI script, maybe same as CERN?)

    Returns
    -------
    alarm_times : list of float
        The times of alarm (predicted disruption)
        If no disruption is predicted, returns None in that position
    """

    # Make array of alarm times the same length as thresholds, starting with all None
    alarm_times = [None] * len(thresholds)

    # Make array of saved times the same length as thresholds
    saved_times = np.zeros(len(thresholds))

    # Go through the shot data
    for i in range(len(risk_at_time)):
        # Get the risk and time
        risk = risk_at_time.iloc[i]['risk']
        time = risk_at_time.iloc[i]['time']

        # Go through the thresholds
        for j in range(len(thresholds)):
            # If alarm has already been triggered, skip
            if alarm_times[j] is not None:
                continue

            # If the risk is above the upper threshold and we don't already have a time saved, save the time
            if risk > thresholds[j][1] and saved_times[j] == 0:
                saved_times[j] = time

            # If the risk is below the lower threshold, reset the saved time
            elif risk < thresholds[j][0]:
                saved_times[j] = 0

            # Check if enough time has elapsed to predict a disruption
            if time - saved_times[j] > thresholds[j][2]:
                # Save alarm time
                alarm_times[j] = time

    return alarm_times

def calculate_alarm_times_ettd(ettd_at_time, thresholds):
    """
    Calculates the alarm times for a given shot using the expected time to disruption.
    If the expected time to disruption output of the model drops below some threshold,
    a disruption is predicted.

    Parameters
    ----------
    ettd_at_time : pandas.DataFrame
        The expected time to disruption for each time slice in a single shot
        Should be sorted by time
    thresholds: list of float
        An alarm is triggered if expected time to disruption < threshold

    Returns
    -------
    alarm_times : list
        The times of alarm for each threshold. 
        If no alarm is triggered, returns None in that position
    """

    # Make array of alarm times the same length as thresholds, starting with all None
    alarm_times = [None] * len(thresholds)

    # Go through the shot data
    for i in range(len(ettd_at_time)):
        # Get the risk and time
        ettd = ettd_at_time.iloc[i]['risk']
        time = ettd_at_time.iloc[i]['time']

        # Go through the thresholds
        for j in range(len(thresholds)):
            # If alarm has already been triggered, skip
            if alarm_times[j] is not None:
                continue
            
            # If the ettd is below the threshold, save the alarm time
            if ettd < thresholds[j]:
                alarm_times[j] = time

    return alarm_times

# Calculating evaluation metrics

def timeslice_micro_average(device, dataset_path, model, experiment_type):
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
    x_vals = x_vals[x_vals < x_cutoff]
    y_vals = y_vals[x_vals < x_cutoff]

    # Sort the values by x
    x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals)))

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

def expected_time_to_disruption_integral():
    """ Calculate the integral of the difference between the expected time to disruption and the actual time to disruption,
            for a given horizon and required warning time
            
            This implementation will need to heavily weight the shots that disrupted.
            """
    pass

# Other functions

def clump_many_to_one_statistics(unique_values_raw, clumping_values, epsilon=0.01):
    """
    For example, The way this is calculated, the warning times and true alarm rates and false alarm rates are all given by particular thresholds
    As such, we can easily compare them to the thresholds, since each value corresponds to one threshold
    However, when comparing them to eachother, this becomes difficult because there is not necessarily a one-to-one correspondence
    For instance, we could have a true alarm rate of 0.5 for both a threshold of 0.1 and 0.2, but the warning times could be different
    This function clumps values together for each unique value

    Parameters
    ----------
    unique_values_raw : numpy.ndarray
        The unique values which dictate the clumping. Not necessarily unique at this point.
    clumping_values : numpy.ndarray
        The values to be clumped. Must be the same length as unique_values.
    epsilon : float
        The maximum difference between two unique values to be considered the same

    Returns
    -------
    unique_true_alarm_rates : numpy.ndarray
        The unique true alarm rates
    avg_warning_times : numpy.ndarray
        The average clumped value for each unique value
    std_warning_times : numpy.ndarray
        The standard deviation of the clumped values for each unique value
    
    """

    # Within these unique values, if they are within epsilon of eachother, eliminate one
    # TODO

    # Actually trim down to the unique values
    try:
        unique_values = np.unique(unique_values_raw, axis=0)
    except:
        unique_values = np.unique(unique_values_raw)

    # Initialize the average and standard deviation arrays
    avg_clump_values = np.zeros(len(unique_values))
    std_clump_values = np.zeros(len(unique_values))

    # Go through each unique value and find the clumping values which correspond to it
    for i, unique_value in enumerate(unique_values):
        # Find the indices of the raw unqiue values that correspond to this particular unique value
        indices = np.where(unique_values_raw == unique_value)

        # Get the clumping values that correspond to this unique value
        # TODO: polish up this list comprehension to be more readable
        try:
            clumping_values_2D = [clumping_values[k][j] for k in range(len(clumping_values)) for j in indices]
            # Flatten the list
            clumping_values_1D = [item for sublist in clumping_values_2D for item in sublist]
        except:
            clumping_values_1D = [clumping_values[j] for j in indices]

        # Calculate the average and standard deviation of the clumped values
        avg_clump_values[i] = np.mean(clumping_values_1D)
        std_clump_values[i] = np.std(clumping_values_1D)

    return unique_values, avg_clump_values, std_clump_values
