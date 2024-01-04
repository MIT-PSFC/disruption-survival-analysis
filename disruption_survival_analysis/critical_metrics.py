# Methods for computing the metrics which are absolutely critical for the project

import numpy as np
from disruption_survival_analysis.DisruptionPredictors import MAX_FUTURE_LIFETIME

def compute_metrics_vs_risk_thresholds(predictions, outcomes, required_warning_time, thresholds):
    """ Compute the true alarm rate, false alarm rate, and average/standard deviation of warning time
    at each risk threshold for a given set of predictions and outcomes.
    A true alarm is defined as an alarm that is triggered with warning time greater than the required warning
    time on a disruptive shots.
    For disruptive shots where an alarm is never triggered, the warning time is 0.
        
    Parameters
    ----------
    predictions : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'risk': a numpy array of risk values
            'time': a numpy array of time values
    outcomes : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'disruption_time': float. the time of disruption. If the shot did not disrupt, this value is np.NaN
            'disrupted': bool. whether or not the shot actually disrupted
    required_warning_time : float
        The minimum warning time required for an alarm to be considered a true alarm
    thresholds : numpy array
        The thresholds to compute the metrics at. Array of floats between 0 and 1, sorted from lowest to highest.
    
    Returns
    -------
    true_alarm_rates : numpy array
        The true alarm rates corresponding to each threshold
    false_alarm_rates : numpy array
        The false alarm rates corresponding to each threshold
    avg_warning_times : numpy array
        The average warning times for the given predictions and true outcomes
    
    """
    true_positives = np.zeros(len(thresholds))
    false_positives = np.zeros(len(thresholds))
    total_warning_time = np.zeros(len(thresholds))
    warning_times = np.zeros((len(predictions), len(thresholds)))
    disruptive_shot_mask = np.zeros(len(predictions))

    for i in range(len(predictions)):
        risk_values = predictions[i]['risk']
        time_values = predictions[i]['time']
        disruption_time = outcomes[i]['disruption_time']
        disrupted = outcomes[i]['disrupted']
        
        # Each row of alarms_triggered corresponds to a threshold
        # Each column of alarms_triggered corresponds to a time
        # For a risk-based alarm, an alarm is triggered when the risk exceeds the threshold value for the first time
        alarms_triggered = (risk_values[:, np.newaxis] > thresholds).T.astype(int)

        if disrupted:
            alarm_times = np.where(alarms_triggered==1, time_values[:, np.newaxis].T, np.inf)
            first_alarm_times = alarm_times.min(axis=1)

            warning_time = np.maximum(0, disruption_time - first_alarm_times)

            true_positives += (warning_time > required_warning_time).astype(int)
            
            total_warning_time += warning_time
            warning_times[i] = warning_time
            
            disruptive_shot_mask[i] = 1
        else:
            false_positives += alarms_triggered.any(axis=1).astype(int)

    disruptive_shots = np.sum(disruptive_shot_mask)
    non_disruptive_shots = len(predictions) - disruptive_shots
    true_alarm_rates = true_positives / disruptive_shots
    false_alarm_rates = false_positives / non_disruptive_shots
    avg_warning_times = total_warning_time / disruptive_shots
    
    return true_alarm_rates, false_alarm_rates, avg_warning_times

def compute_metrics_vs_time_thresholds(predictions, outcomes, required_warning_time, thresholds):
    """ Compute the true alarm rate, false alarm rate, and average/standard deviation of warning time
    at each time threshold for a given set of predictions and outcomes.
    A true alarm is defined as an alarm that is triggered with warning time greater than the required warning
    time on a disruptive shots.
    For disruptive shots where an alarm is never triggered, the warning time is 0.
        
    Parameters
    ----------
    predictions : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'ettd': a numpy array of expected time to disruption values
            'time': a numpy array of time values
    outcomes : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'disruption_time': float. the time of disruption. If the shot did not disrupt, this value is np.NaN
            'disrupted': bool. whether or not the shot actually disrupted
    required_warning_time : float
        The minimum warning time required for an alarm to be considered a true alarm
    thresholds : numpy array
        The thresholds to compute the metrics at. Array of floats between 0 and 1, sorted from lowest to highest.
    
    Returns
    -------
    true_alarm_rates : numpy array
        The true alarm rates corresponding to each threshold
    false_alarm_rates : numpy array
        The false alarm rates corresponding to each threshold
    avg_warning_times : numpy array
        The average warning times for the given predictions and true outcomes
    std_warning_times : numpy array
        The standard deviation of the warning times for the given predictions and true outcomes
    
    """
    true_positives = np.zeros(len(thresholds))
    false_positives = np.zeros(len(thresholds))
    total_warning_time = np.zeros(len(thresholds))
    warning_times = np.zeros((len(predictions), len(thresholds)))
    disruptive_shots = 0
    non_disruptive_shots = 0

    for i in range(len(predictions)):
        ettd_values = predictions[i]['ettd']
        time_values = predictions[i]['time']
        disruption_time = outcomes[i]['disruption_time']
        disrupted = outcomes[i]['disrupted']
        
        # Each row of alarms_triggered corresponds to a threshold
        # Each column of alarms_triggered corresponds to a time
        # For a time-based alarm, an alarm is triggered when the expected time drops below the threshold
        alarms_triggered = (ettd_values[:, np.newaxis] <= thresholds).T.astype(int)

        if disrupted:
            alarm_times = np.where(alarms_triggered==1, time_values[:, np.newaxis].T, np.inf)
            first_alarm_times = alarm_times.min(axis=1)

            warning_time = np.maximum(0, disruption_time - first_alarm_times)

            true_positives += (warning_time > required_warning_time)
            total_warning_time += warning_time
            warning_times[i] = warning_time
            disruptive_shots += 1
        else:
            false_positives += alarms_triggered.any(axis=1).astype(int)
            non_disruptive_shots += 1

    true_alarm_rates = true_positives / disruptive_shots
    false_alarm_rates = false_positives / non_disruptive_shots
    avg_warning_times = total_warning_time / disruptive_shots
    std_warning_times = np.std(warning_times[:disruptive_shots], axis=0)
    
    return true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times

def compute_metrics_vs_false_alarm_rates(predictions, outcomes, required_warning_time, thresholds, threshold_type):
    """ Compute the true alarm rate and average/standard deviation of warning time
    for each unique false alarm rate for a given set of predictions and true outcomes.
    A true alarm is defined as an alarm that is triggered with warning time greater than the required warning
    time on a disruptive shots.
    For disruptive shots where an alarm is never triggered, the warning time is 0.
        
    Parameters
    ----------
    predictions : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'risk': a numpy array of risk values
            'time': a numpy array of time values
    outcomes : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'disruption_time': float. the time of disruption. If the shot did not disrupt, this value is np.NaN
            'disrupted': bool. whether or not the shot actually disrupted
    required_warning_time : float
        The minimum warning time required for an alarm to be considered a true alarm
    thresholds : numpy array
        The thresholds used to trigger alarms. Array of floats between 0 and 1, sorted from lowest to highest.
    
    Returns
    -------
    unique_false_alarm_rates : numpy array
        The unique false alarm rates
    true_alarm_rates : numpy array
        The true alarm rates corresponding to each unique false alarm rate
    avg_warning_times : numpy array
        The average warning times corresponding to each unique false alarm rate
    """

    if threshold_type == 'sthr':
        threshold_true_alarm_rates, threshold_false_alarm_rates, threshold_avg_warning_times = compute_metrics_vs_risk_thresholds(predictions, outcomes, required_warning_time, thresholds)
    elif threshold_type == 'ettd':
        threshold_true_alarm_rates, threshold_false_alarm_rates, threshold_avg_warning_times, threshold_std_warning_times = compute_metrics_vs_time_thresholds(predictions, outcomes, required_warning_time, thresholds)

    # 1. Get the unique false alarm rates from the threshold false alarm rates
    unique_false_alarm_rates = np.unique(threshold_false_alarm_rates)

    # 2. For each unique false alarm rate, compute the average true alarm rate and average warning time
    true_alarm_rates = np.zeros(len(unique_false_alarm_rates))
    avg_warning_times = np.zeros(len(unique_false_alarm_rates))

    # 3. For each unique false alarm rate, compute the average true alarm rate and average warning time
    for i in range(len(unique_false_alarm_rates)):
        false_alarm_rate = unique_false_alarm_rates[i]
        chosen_tars = threshold_true_alarm_rates[threshold_false_alarm_rates == false_alarm_rate]
        true_alarm_rates[i] = np.mean(chosen_tars)
        chosen_avg_warns = threshold_avg_warning_times[threshold_false_alarm_rates == false_alarm_rate]
        avg_warning_times[i] = np.mean(chosen_avg_warns)
    
    # 4. Return the false alarm rates and average warning times
    return unique_false_alarm_rates, true_alarm_rates, avg_warning_times

def compute_warns_vs_risk_thresholds(predictions, outcomes, required_warning_time, thresholds):
    """ Compute the true alarm rate, false alarm rate, and average/standard deviation of warning time
    at each risk threshold for a given set of predictions and outcomes.
    A true alarm is defined as an alarm that is triggered with warning time greater than the required warning
    time on a disruptive shots.
    For disruptive shots where an alarm is never triggered, the warning time is 0.
        
    Parameters
    ----------
    predictions : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'risk': a numpy array of risk values
            'time': a numpy array of time values
    outcomes : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'disruption_time': float. the time of disruption. If the shot did not disrupt, this value is np.NaN
            'disrupted': bool. whether or not the shot actually disrupted
    required_warning_time : float
        The minimum warning time required for an alarm to be considered a true alarm
    thresholds : numpy array
        The thresholds to compute the metrics at. Array of floats between 0 and 1, sorted from lowest to highest.
    
    Returns
    -------
    true_alarm_rates : numpy array
        The true alarm rates corresponding to each threshold
    false_alarm_rates : numpy array
        The false alarm rates corresponding to each threshold
    only_disruptive_warning_times : numpy array
        The warning times for the given predictions and true outcomes on disruptive shots only
    
    """
    true_positives = np.zeros(len(thresholds))
    false_positives = np.zeros(len(thresholds))
    warning_times = np.zeros((len(predictions), len(thresholds)))
    disruptive_shot_mask = np.zeros(len(predictions))

    for i in range(len(predictions)):
        risk_values = predictions[i]['risk']
        time_values = predictions[i]['time']
        disruption_time = outcomes[i]['disruption_time']
        disrupted = outcomes[i]['disrupted']
        
        # Each row of alarms_triggered corresponds to a threshold
        # Each column of alarms_triggered corresponds to a time
        # For a risk-based alarm, an alarm is triggered when the risk exceeds the threshold value for the first time
        alarms_triggered = (risk_values[:, np.newaxis] > thresholds).T.astype(int)

        if disrupted:
            alarm_times = np.where(alarms_triggered==1, time_values[:, np.newaxis].T, np.inf)
            first_alarm_times = alarm_times.min(axis=1)

            warning_time = np.maximum(0, disruption_time - first_alarm_times)

            true_positives += (warning_time > required_warning_time).astype(int)
            
            warning_times[i] = warning_time

            disruptive_shot_mask[i] = 1
        else:
            false_positives += alarms_triggered.any(axis=1).astype(int)

    disruptive_shots = np.sum(disruptive_shot_mask)
    non_disruptive_shots = len(predictions) - disruptive_shots
    true_alarm_rates = true_positives / disruptive_shots
    false_alarm_rates = false_positives / non_disruptive_shots

    only_disruptive_warning_times = warning_times[disruptive_shot_mask == 1]
    
    return true_alarm_rates, false_alarm_rates, only_disruptive_warning_times

def compute_metrics_vs_false_alarm_rates_distribution(predictions, outcomes, required_warning_time, thresholds, threshold_type, requested_metrics):
    """ Compute the true alarm rate and average/standard deviation of warning time
    for each unique false alarm rate for a given set of predictions and true outcomes.
    A true alarm is defined as an alarm that is triggered with warning time greater than the required warning
    time on a disruptive shots.
    For disruptive shots where an alarm is never triggered, the warning time is 0.
        
    Parameters
    ----------
    predictions : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'risk': a numpy array of risk values
            'time': a numpy array of time values
    outcomes : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'disruption_time': float. the time of disruption. If the shot did not disrupt, this value is np.NaN
            'disrupted': bool. whether or not the shot actually disrupted
    required_warning_time : float
        The minimum warning time required for an alarm to be considered a true alarm
    thresholds : numpy array
        The thresholds used to trigger alarms. Array of floats between 0 and 1, sorted from lowest to highest.
    
    Returns
    -------
    unique_false_alarm_rates : numpy array
        The unique false alarm rates
    tar_metrics : dict of numpy arrays
        The true alarm rates corresponding to each unique false alarm rate
    warn_metrics : dict of numpy arrays
        The average warning times corresponding to each unique false alarm rate
    """

    threshold_true_alarm_rates, threshold_false_alarm_rates, threshold_warning_times = compute_warns_vs_risk_thresholds(predictions, outcomes, required_warning_time, thresholds)

    # 1. Get the unique false alarm rates from the threshold false alarm rates
    unique_false_alarm_rates = np.unique(threshold_false_alarm_rates)

    # 2. Set up arrays for the requested metrics to be calculated
    tar_metrics = {}
    warn_metrics = {}
    for metric in requested_metrics:
        if metric not in ['avg', 'std', 'med', 'iq1', 'iq3', 'iqm', 'all']:
            raise ValueError(f"Metric {metric} not recognized")
        if metric == 'all':
            tar_metrics[metric] = []
            warn_metrics[metric] = []
        else:
            tar_metrics[metric] = np.zeros(len(unique_false_alarm_rates))
            warn_metrics[metric] = np.zeros(len(unique_false_alarm_rates))

    for i in range(len(unique_false_alarm_rates)):
        false_alarm_rate = unique_false_alarm_rates[i]

        chosen_trues = threshold_true_alarm_rates[threshold_false_alarm_rates == false_alarm_rate]
        chosen_warns = threshold_warning_times[:,threshold_false_alarm_rates == false_alarm_rate]

        if 'avg' in requested_metrics:
            tar_metrics['avg'][i] = np.mean(chosen_trues)
            warn_metrics['avg'][i] = np.mean(chosen_warns)
        if 'std' in requested_metrics:
            tar_metrics['std'][i] = np.std(chosen_trues)
            warn_metrics['std'][i] = np.std(chosen_warns)
        if 'med' in requested_metrics:
            tar_metrics['med'][i] = np.median(chosen_trues)
            warn_metrics['med'][i] = np.median(chosen_warns)
        if 'iq1' in requested_metrics:
            tar_metrics['iq1'][i] = np.quantile(chosen_trues, 0.25)
            warn_metrics['iq1'][i] = np.quantile(chosen_warns, 0.25)
        if 'iq3' in requested_metrics:
            tar_metrics['iq3'][i] = np.quantile(chosen_trues, 0.75)
            warn_metrics['iq3'][i] = np.quantile(chosen_warns, 0.75)
        if 'iqm' in requested_metrics:
            warn_metrics['iqm'][i] = interquartile_mean(chosen_warns.flatten())
            
        if 'all' in requested_metrics:
            warn_metrics['all'].append(chosen_warns)

    # 4. Return the false alarm rates and average warning times
    return unique_false_alarm_rates, tar_metrics, warn_metrics

def interquartile_mean(values):
    """Calculate the interquartile mean of a list of values."""
    sorted_values = np.sort(values)

    if len(sorted_values) < 3:
        iqm = np.mean(sorted_values)
    else:
        size = len(sorted_values)/4
        iqm = np.mean(sorted_values[int(size):int(size*3)])

    return iqm

def compute_simple_rmst_integral(rmst, times, disruptive):
    """Compute the simple RMST integral for a predicted array of RMST values.
    The simple RMST integral is defined as the integrated absolute difference between the
    perfect RMST (line with slope -1 for disruptive shots, flat line at MAX_FUTURE_LIFETIME for non-disruptive shots)
    
    Parameters
    ----------
    rmst : numpy array
        The predicted RMST values for each time slice in the shot
    times : numpy array
        The times corresponding to each RMST value
    disruptive : boolean
        A boolean array indicating whether or not the shot actually disrupted
    
    Returns
    -------
    simple_rmst_integral : float
        The simple RMST integral
    """

    if disruptive:
        # Perfect rmst is a line with slope -1 that intersects 0 at the disruption time
        # And flat at MAX_FUTURE_LIFETIME before then
        perfect_rmst = np.minimum(times[0] - times + times[-1], MAX_FUTURE_LIFETIME)
    else:
        perfect_rmst = np.ones(len(times)) * MAX_FUTURE_LIFETIME

    simple_rmst_integral = np.trapz(np.abs(rmst - perfect_rmst), times)

    return simple_rmst_integral