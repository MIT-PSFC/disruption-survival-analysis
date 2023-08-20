import numpy as np

def compute_metric(predictions:list(list), true_outcomes):
    """
    
    Paramteres:
    ------
    predictions:
        prediction of risk at a certain horizon for each shot
    true_outcomes:
        List of signal values over time 
        list of 7000 shots
        - for each list item, there is the disruption time and whether or not the shot disrupted
    """

    all_predictions = []
    for prediction in predictions:
        all_predictions = all_predictions + prediction

    all_predictions = list(set(all_predictions))

    all_thresholds = []
    all_false_alarm_rates = []
    all_avg_warning_times = []

    for threshold in all_predictions:
        alarms = 0
        true_alarms = 0

        warning_times = []

        for i, prediction in enumerate(predictions):
            if (prediction > threshold).any():
                alarms += 1
                if true_outcomes[i] == True:
                    true_alarms += 1
                warning_time = np.where(predictions > threshold)[0] * SAMPLE_FREQUENCY

                warning_times.append(warning_time)

        false_alarms = alarms - true_alarms
        false_alarm_rate = false_alarms / len(predictions)

        all_thresholds.append(threshold)
        all_false_alarm_rates.append(false_alarm_rate)

        avg_warning_time = np.mean(warning_times)

        all_avg_warning_times.append(avg_warning_time)

    return all_thresholds, all_false_alarm_rates, all_avg_warning_times