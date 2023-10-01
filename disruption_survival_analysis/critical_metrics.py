import numpy as np

SIMPLE_THRESHOLDS = np.linspace(0, 1, 100)

def compute_avg_warning_time_vs_false_alarm_rate(predictions, true_outcomes, required_warning_time, thresholds):
    """ Compute the critical metric for a given set of predictions and true outcomes.
        The critical metric is the average warning time for a given false alarm rate.

    Parameters
    ----------
    predictions : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'risk': a numpy array of risk values
            'time': a numpy array of time values
    true_outcomes : list of dictionaries
        Each dictionary corresponds to a single shot and contains the following keys:
            'disruption_time': the time of disruption
            'disrupted': whether or not the shot actually disrupted
    
    Returns
    -------
    false_alarm_rates : numpy array
        The false alarm rates for the given predictions and true outcomes
    avg_warning_times : numpy array
        The average warning times for the given predictions and true outcomes
    std_warning_times : numpy array
        The standard deviation of the warning times for the given predictions and true outcomes
    
    """


def compute_metrics_vs_thresholds(predictions, outcomes, required_warning_time, thresholds):
    num_negatives = sum(1 for outcome in outcomes if not outcome['disrupted'])
    num_positives = len(outcomes) - num_negatives

    risks = [pred['risk'] for pred in predictions]
    times = [pred['time'] for pred in predictions]
    disruption_times = [outcome['disruption_time'] for outcome in outcomes]
    disrupted = [outcome['disrupted'] for outcome in outcomes]

    # Convert lists to arrays for faster computation
    risks = np.array(risks)
    times = np.array(times)
    disruption_times = np.array(disruption_times)
    disrupted = np.array(disrupted)

    alarms_triggered = risks[:, :, None] > thresholds
    warning_times = disruption_times[:, None] - times[:, :, None]

    valid_alarms = (warning_times > required_warning_time) & alarms_triggered & disrupted[:, None, None]
    false_alarms = alarms_triggered & ~disrupted[:, None, None]

    true_alarms_per_threshold = valid_alarms.sum(axis=0)
    false_alarms_per_threshold = false_alarms.sum(axis=0)

    true_alarm_rates = true_alarms_per_threshold / num_positives
    false_alarm_rates = false_alarms_per_threshold / num_negatives

    avg_warning_times = np.where(valid_alarms, warning_times, 0).sum(axis=0) / true_alarms_per_threshold
    avg_warning_times = np.nan_to_num(avg_warning_times)  # handle cases where true_alarms_per_threshold = 0

    std_warning_times = np.sqrt(((np.where(valid_alarms, warning_times - avg_warning_times, 0) ** 2).sum(axis=0)) / true_alarms_per_threshold)
    std_warning_times = np.nan_to_num(std_warning_times)  # handle cases where true_alarms_per_threshold = 0

    return true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times


# def compute_metrics_vs_thresholds(predictions, outcomes, required_warning_time, thresholds):
#     """ Compute the true alarm rate, false alarm rate, and average/standard deviation of warning time
#     at each threshold for a given set of predictions and outcomes.
#     A true alarm is defined as an alarm that is triggered with warning time greater than the required warning
#     time on a disruptive shots.
#     For disruptive shots where an alarm is never triggered, the warning time is 0.
        
#     Parameters
#     ----------
#     predictions : list of dictionaries
#         Each dictionary corresponds to a single shot and contains the following keys:
#             'risk': a numpy array of risk values
#             'time': a numpy array of time values
#     outcomes : list of dictionaries
#         Each dictionary corresponds to a single shot and contains the following keys:
#             'disruption_time': float. the time of disruption. If the shot did not disrupt, this value is np.NaN
#             'disrupted': bool. whether or not the shot actually disrupted
#     required_warning_time : float
#         The minimum warning time required for an alarm to be considered a true alarm
#     thresholds : numpy array
#         The thresholds to compute the metrics at. Array of floats between 0 and 1, sorted from lowest to highest.
    
#     Returns
#     -------
#     true_alarm_rates : numpy array
#         The true alarm rates corresponding to each threshold
#     false_alarm_rates : numpy array
#         The false alarm rates corresponding to each threshold
#     avg_warning_times : numpy array
#         The average warning times corresponding to each threshold
#     std_warning_times : numpy array
#         The standard deviation of the warning times corresponding to each threshold
    
#     """
#     # Count the number of negatives and positives in the outcomes
#     num_negatives = 0
#     num_positives = 0
#     for outcome in outcomes:
#         if outcome['disrupted']:
#             num_positives += 1
#         else:
#             num_negatives += 1

#     # For each unique predicted risk, find the alarm rates and average warning time
#     # Average warning time is only defined for disruptive shots

#     num_thresholds = len(thresholds)
#     true_alarm_rates = np.zeros(num_thresholds)
#     false_alarm_rates = np.zeros(num_thresholds)
#     avg_warning_times = np.zeros(num_thresholds)
#     std_warning_times = np.zeros(num_thresholds)

#     for i, threshold in enumerate(thresholds):
#         alarms = 0
#         true_alarms = 0
#         warning_times = []

#         # Iterate through each shot
#         for j, prediction in enumerate(predictions):
#             # Iterate through each predicted risk at time in the shot.
#             # Once an alarm is triggered, this loop gets broken out of, 
#             # because only one alarm can be triggered per shot at a given threshold.
#             warning_time = None
#             for k, risk in enumerate(prediction['risk']):
#                 if risk > threshold:
#                     # The risk exceeded the threshold. Determine if it was a true alarm.
#                     if outcomes[j]['disrupted']:
#                         # Shot was disruptive. Determine if the alarm was triggered in time
#                         warning_time = outcomes[j]['disruption_time'] - prediction['time'][k]
#                         if warning_time > required_warning_time:
#                             # Alarm was triggered in time on a disruptive shot. True alarm.
#                             alarms += 1
#                             true_alarms += 1
#                             warning_times.append(warning_time)
#                     else:
#                         # Shot was not disruptive. False alarm.
#                         alarms += 1
#                     break
#             # Ran through all risks in the shot and none exceeded the threshold. No alarm.
#             # If the shot was disruptive, the warning time is 0.
#             if outcomes[j]['disrupted'] and warning_time is None:
#                 warning_times.append(0)

#         # Copmpute statistics for this threshold
#         true_alarm_rate = true_alarms / num_positives

#         false_alarms = alarms - true_alarms
#         false_alarm_rate = false_alarms / num_negatives

#         avg_warning_time = np.mean(warning_times)
#         std_warning_time = np.std(warning_times)

#         true_alarm_rates[i] = true_alarm_rate
#         false_alarm_rates[i] = false_alarm_rate
#         avg_warning_times[i] = avg_warning_time
#         std_warning_times[i] = std_warning_time

#     return true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times

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
    true_alarm_rates : numpy array
        The true alarm rates corresponding to each unique false alarm rate
    avg_warning_times : numpy array
        The average warning times corresponding to each unique false alarm rate
    std_warning_times : numpy array
        The standard deviation of the warning times corresponding to each unique false alarm rate
    
    """
    # 1. Set Up

    # Count the number of negatives in the true outcomes
    num_negatives = 0
    for true_outcome in outcomes:
        if not true_outcome['disrupted']:
            num_negatives += 1

    # 2. For each unique predicted risk, find the false alarm rate and average warning time
    # Average warning time is only defined for disruptive shots

    all_false_alarm_rates = []

    true_alarms = []
    all_warning_times = []

    for threshold in thresholds:
        alarms = 0
        true_alarms = 0
        warning_times = []

        # Iterate through each shot
        for i, prediction in enumerate(predictions):
            # Iterate through each predicted risk at time in the shot.
            # Once an alarm is triggered, this loop gets broken out of, 
            # because only one alarm can be triggered per shot at a given threshold.
            warning_time = None
            for j, risk in enumerate(prediction['risk']):
                if risk > threshold:
                    # The risk exceeded the threshold. Determine if it was a true alarm.
                    if outcomes[i]['disrupted']:
                        # Shot was disruptive. Determine if the alarm was triggered in time
                        warning_time = outcomes[i]['disruption_time'] - prediction['time'][j]
                        if warning_time > required_warning_time:
                            # Alarm was triggered in time on a disruptive shot. True alarm.
                            alarms += 1
                            true_alarms += 1
                            warning_times.append(warning_time)
                    else:
                        # Shot was not disruptive. False alarm.
                        alarms += 1
                    break
            # Ran through all risks in the shot and none exceeded the threshold. No alarm.
            # If the shot was disruptive, the warning time is 0.
            if outcomes[i]['disrupted'] and warning_time is None:
                warning_times.append(0)

        # Compute the false alarm rate and average warning time
        false_alarms = alarms - true_alarms
        false_alarm_rate = false_alarms / num_negatives

        # Add to the list of all alarms, false alarm rates, and list of warning times
        all_false_alarm_rates.append(false_alarm_rate)
        all_warning_times.append(warning_times)

    # 3. Make it so the false alarm rates are unique
    # Since false alarm rate depends on the data of non-disruptive shots, 
    # while warning time depends on the data of disruptive shots,
    # the false alarm rate can be the same for multiple thresholds, while the warning time is different.
    # This means a single value in the domain maps to multiple values in the range, which is not a function
    # To fix this, we will average the warning times for each unique false alarm rate

    unique_false_alarm_rates = np.unique(all_false_alarm_rates)
    avg_true_alarm_rates = []
    avg_warning_times = []
    std_warning_times = []
    
    for false_alarm_rate in unique_false_alarm_rates:
        # all_warning_times is a list of lists, where each list is the warning times for a given false alarm rate
        # for each unique false alarm rate, find the corresponding warning times and average them
        warning_times = []
        true_alarm_rates = []
        for i, _ in enumerate(all_false_alarm_rates):
            if all_false_alarm_rates[i] == false_alarm_rate:
                warning_times.extend(all_warning_times[i])
        # Compute averages and standard deviations
        avg_warning_time = np.mean(warning_times)
        std_warning_time = np.std(warning_times)
        # If the grouped values are empty, set the average and standard deviation to 0
        if np.isnan(avg_warning_time):
            avg_warning_time = 0
        if np.isnan(std_warning_time):
            std_warning_time = 0
        # Average the warning times
        avg_warning_times.append(avg_warning_time)
        std_warning_times.append(std_warning_time)

    # 4. Return the false alarm rates and average warning times
    return unique_false_alarm_rates, avg_warning_times, std_warning_times




# def compute_metric(predictions:list(list), true_outcomes):
#     """
    
#     Paramteres:
#     ------
#     predictions:
#         prediction of risk at a certain horizon for each shot
#     true_outcomes:
#         List of signal values over time 
#         list of 7000 shots
#         - for each list item, there is the disruption time and whether or not the shot disrupted
#     """

#     all_predictions = []
#     for prediction in predictions:
#         all_predictions = all_predictions + prediction

#     all_predictions = list(set(all_predictions))

#     all_thresholds = []
#     all_false_alarm_rates = []
#     all_avg_warning_times = []

#     for threshold in all_predictions:
#         alarms = 0
#         true_alarms = 0

#         warning_times = []

#         for i, prediction in enumerate(predictions):
#             if (prediction > threshold).any():
#                 alarms += 1
#                 if true_outcomes[i] == True:
#                     true_alarms += 1
#                 warning_time = np.where(predictions > threshold)[0] * SAMPLE_FREQUENCY

#                 warning_times.append(warning_time)

#         false_alarms = alarms - true_alarms
#         false_alarm_rate = false_alarms / len(predictions)

#         all_thresholds.append(threshold)
#         all_false_alarm_rates.append(false_alarm_rate)

#         avg_warning_time = np.mean(warning_times)

#         all_avg_warning_times.append(avg_warning_time)

#     return all_thresholds, all_false_alarm_rates, all_avg_warning_times

from joblib import Parallel, delayed

def compute_metrics_for_threshold(risks, times, disruption_times, disrupted, threshold, required_warning_time, num_positives, num_negatives):
    alarms_triggered = risks > threshold
    warning_times = disruption_times[:, None] - times

    valid_alarms = (warning_times > required_warning_time) & alarms_triggered & disrupted[:, None]
    false_alarms = alarms_triggered & ~disrupted[:, None]

    true_alarms_per_threshold = valid_alarms.sum(axis=0)
    false_alarms_per_threshold = false_alarms.sum(axis=0)

    true_alarm_rate = true_alarms_per_threshold / num_positives
    false_alarm_rate = false_alarms_per_threshold / num_negatives

    avg_warning_time = np.where(valid_alarms, warning_times, 0).mean(axis=0)
    std_warning_time = np.where(valid_alarms, warning_times - avg_warning_time, 0).std(axis=0)

    return true_alarm_rate, false_alarm_rate, avg_warning_time, std_warning_time

def compute_metrics_vs_thresholds_parallel(predictions, outcomes, required_warning_time, thresholds):
    num_negatives = sum(1 for outcome in outcomes if not outcome['disrupted'])
    num_positives = len(outcomes) - num_negatives

    risks = np.array([pred['risk'] for pred in predictions])
    times = np.array([pred['time'] for pred in predictions])
    disruption_times = np.array([outcome['disruption_time'] for outcome in outcomes])
    disrupted = np.array([outcome['disrupted'] for outcome in outcomes])

    results = Parallel(n_jobs=-1)(delayed(compute_metrics_for_threshold)(risks, times, disruption_times, disrupted, threshold, required_warning_time, num_positives, num_negatives) for threshold in thresholds)

    true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times = zip(*results)
    
    return true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times