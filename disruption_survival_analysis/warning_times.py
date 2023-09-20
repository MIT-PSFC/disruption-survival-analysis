import numpy as np

SIMPLE_THRESHOLDS = np.linspace(0, 1, 100)

def compute_critical_metric(predictions, true_outcomes, required_warning_time, thresholds):
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

    # 1. Set Up

    # Count the number of negatives in the true outcomes
    num_negatives = 0
    for true_outcome in true_outcomes:
        if not true_outcome['disrupted']:
            num_negatives += 1

    # 2. For each unique predicted risk, find the false alarm rate and average warning time
    # Average warning time is only defined for disruptive shots

    all_false_alarm_rates = []
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
            for j, risk in enumerate(prediction['risk']):
                if risk > threshold:
                    # The risk exceeded the threshold. Determine if it was a true alarm.
                    if true_outcomes[i]['disrupted']:
                        # Shot was disruptive. Determine if the alarm was triggered in time
                        warning_time = true_outcomes[i]['disruption_time'] - prediction['time'][j]
                        if warning_time > required_warning_time:
                            # Alarm was triggered in time on a disruptive shot. True alarm.
                            alarms += 1
                            true_alarms += 1
                            warning_times.append(warning_time)
                        else:
                            # Alarm was triggered too late. Missed alarm.
                            # Do not increment alarms or true_alarms
                            pass
                    else:
                        # Shot was not disruptive. False alarm.
                        alarms += 1
                    break

        # Compute the false alarm rate and average warning time
        false_alarms = alarms - true_alarms
        false_alarm_rate = false_alarms / num_negatives

        # Add to the list of all alarms, false alarm rates, and list of warning times
        all_false_alarm_rates.append(false_alarm_rate)
        all_warning_times.append(warning_times)

    # 3. Make it so the false alarm rates are unique
    # Since false alarm rate depends on the data of non-disruptive shots, 
    # while average warning time depends on the data of disruptive shots,
    # the false alarm rate can be the same for multiple thresholds, while the average warning time is different.
    # This means a single value in the domain maps to multiple values in the range, which is not a function
    # To fix this, we will average the warning times for each unique false alarm rate

    unique_false_alarm_rates = np.unique(all_false_alarm_rates)
    avg_warning_times = []
    std_warning_times = []

    for false_alarm_rate in unique_false_alarm_rates:
        # all_warning_times is a list of lists, where each list is the warning times for a given false alarm rate
        # for each unique false alarm rate, find the corresponding warning times and average them
        warning_times = []
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