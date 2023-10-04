# Methods for computing the metrics which are absolutely critical for the project

import numpy as np

def compute_metrics_vs_thresholds(predictions, outcomes, required_warning_time, thresholds):
    """ Compute the true alarm rate, false alarm rate, and average/standard deviation of warning time
    at each threshold for a given set of predictions and outcomes.
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
        risk_values = predictions[i]['risk']
        time_values = predictions[i]['time']
        disruption_time = outcomes[i]['disruption_time']
        disrupted = outcomes[i]['disrupted']
        
        alarms_triggered = (risk_values[:, np.newaxis] > thresholds).any(axis=0)

        if disrupted:
            if np.any(alarms_triggered):
                first_alarm_time = np.where(alarms_triggered, time_values[:, np.newaxis], np.inf).min(axis=0)
                warning_time = np.maximum(0, disruption_time - first_alarm_time) * disrupted
            else:
                warning_time = np.zeros(len(thresholds))
            true_positives += (warning_time > required_warning_time)
            total_warning_time += warning_time
            warning_times[i] = warning_time
            disruptive_shots += 1
        else:
            false_positives += (alarms_triggered).astype(int)
            non_disruptive_shots += 1

    true_alarm_rates = true_positives / disruptive_shots
    false_alarm_rates = false_positives / non_disruptive_shots
    avg_warning_times = total_warning_time / disruptive_shots
    std_warning_times = np.std(warning_times[:disruptive_shots], axis=0)
    
    return true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times

def compute_metrics_vs_false_alarm_rates(predictions, outcomes, required_warning_time, thresholds):
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
    std_warning_times : numpy array
        The standard deviation of the warning times corresponding to each unique false alarm rate
    
    """

    threshold_true_alarm_rates, threshold_false_alarm_rates, threshold_avg_warning_times, threshold_std_warning_times = compute_metrics_vs_thresholds(predictions, outcomes, required_warning_time, thresholds)

    # 1. Get the unique false alarm rates from the threshold false alarm rates
    unique_false_alarm_rates = np.unique(threshold_false_alarm_rates)

    # 2. For each unique false alarm rate, compute the average true alarm rate and average warning time
    true_alarm_rates = np.zeros(len(unique_false_alarm_rates))
    avg_warning_times = np.zeros(len(unique_false_alarm_rates))
    std_warning_times = np.zeros(len(unique_false_alarm_rates))

    # Compute variance of warning time for each false alarm rate
    threshold_var_warning_times = threshold_std_warning_times ** 2

    # 3. For each unique false alarm rate, compute the average true alarm rate and average warning time
    for i in range(len(unique_false_alarm_rates)):
        false_alarm_rate = unique_false_alarm_rates[i]
        true_alarm_rates[i] = np.mean(threshold_true_alarm_rates[threshold_false_alarm_rates == false_alarm_rate])
        avg_warning_times[i] = np.mean(threshold_avg_warning_times[threshold_false_alarm_rates == false_alarm_rate])
        std_warning_times[i] = np.mean(threshold_var_warning_times[threshold_false_alarm_rates == false_alarm_rate]) ** 0.5
    
    # 4. Return the false alarm rates and average warning times
    return unique_false_alarm_rates, true_alarm_rates, avg_warning_times, std_warning_times

