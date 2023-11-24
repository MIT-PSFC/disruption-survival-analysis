# Methods for computing the metrics which are absolutely critical for the project

import numpy as np

WARNING_TIME_CUTOFF = 0.5 # Ignore warnings that are more than 500ms before disruption

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
        
        # Each row of alarms_triggered corresponds to a threshold
        # Each column of alarms_triggered corresponds to a time
        # For a risk-based alarm, an alarm is triggered when the risk exceeds the threshold value for the first time
        alarms_triggered = (risk_values[:, np.newaxis] > thresholds).T.astype(int)

        if disrupted:
            alarm_times = np.where(alarms_triggered==1, time_values[:, np.newaxis].T, np.inf)
            first_alarm_times = alarm_times.min(axis=1)

            warning_time = np.maximum(0, disruption_time - first_alarm_times)

            true_positives += ((warning_time > required_warning_time) * (warning_time <= WARNING_TIME_CUTOFF))
            
            false_positives += (warning_time > WARNING_TIME_CUTOFF).astype(int)
            
            total_warning_time += warning_time
            warning_times[i] = warning_time

            # For a shot where a disruption occurs, we are basically splitting it into
            # a non-disruptive shot and a disruptive shot (if it is long enough)
            # This allows us to count early warnings as a false positive
            # without letting the false positive rate go above 1
            shot_duration = time_values[-1] - time_values[0]
            if shot_duration > WARNING_TIME_CUTOFF:
                non_disruptive_shots += 1
            
            disruptive_shots += 1
        else:
            false_positives += alarms_triggered.any(axis=1).astype(int)
            non_disruptive_shots += 1

    true_alarm_rates = true_positives / disruptive_shots
    false_alarm_rates = false_positives / non_disruptive_shots
    avg_warning_times = total_warning_time / disruptive_shots
    std_warning_times = np.std(warning_times[:disruptive_shots], axis=0)
    
    return true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times

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
    std_warning_times : numpy array
        The standard deviation of the warning times corresponding to each unique false alarm rate
    """

    if threshold_type == 'sthr':
        threshold_true_alarm_rates, threshold_false_alarm_rates, threshold_avg_warning_times, threshold_std_warning_times = compute_metrics_vs_risk_thresholds(predictions, outcomes, required_warning_time, thresholds)
    elif threshold_type == 'ettd':
        threshold_true_alarm_rates, threshold_false_alarm_rates, threshold_avg_warning_times, threshold_std_warning_times = compute_metrics_vs_time_thresholds(predictions, outcomes, required_warning_time, thresholds)

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

