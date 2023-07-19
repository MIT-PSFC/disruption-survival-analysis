"""Utilities to evaluate the performance of a disruption predictor"""

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from DisruptionPredictors import DisruptionPredictor

def benchmark_au_roc(predictor:DisruptionPredictor, horizons, device, dataset):
    """
    Calculate the area under the ROC curve for a given predictor
    at a variety of time horizons

    Parameters
    ----------
    predictor : DisruptionPredictor
        The predictor to evaluate
    horizons : list of float
        How far into the future the predictor is looking
    device : str
        The device to evaluate the predictor on
    dataset : str
        The dataset to evaluate the predictor on
        Should be a similar dataset to the one the predictor was trained on
        and that the predictor's transformer was calculated for
    Returns:
    --------
    au_rocs : list of float
        The area under the ROC curve for each horizon
    """

    # Load the data and process it with predictor's transformer
    data = load_benchmark_data(predictor, device, dataset)

    # TODO: calculating all horizons at once would be more efficient, but not working yet
    #au_rocs = calc_au_roc(predictor, horizons, data)

    # Calculate area under the ROC curve for all horizons
    au_rocs = []
    for horizon in horizons:
        y_true = []
        y_score = []
        for entry in data:
            disrupt = entry[0]
            shot_data = entry[1]

            # Label the data as disruptive or not disruptive at a given horizon
            labeled_data = label_shot_data(shot_data, disrupt, horizon)

            # Find predicted risk for each time slice
            scored_data = predictor.calculate_risk(shot_data, horizon)['risk'].values
            
            y_true = np.concatenate((y_true, labeled_data), axis=None)
            y_score = np.concatenate((y_score, scored_data), axis=None)

        # TODO: There are MANY more 'not disruptive' shots than 'disruptive' time slices
        # Don't do weighting though, acc Chirag
        # Doesn't appear to make much of a difference.
        # Calculate the area under the ROC curve 
        au_roc = roc_auc_score(y_true, y_score)

        #au_roc = calc_au_roc(predictor, horizon, data)
        au_rocs.append(au_roc)

    return au_rocs



def calc_au_roc(predictor:DisruptionPredictor, horizon, data):
    """
    DEPRECATED. USING SCIKIT-LEARN INSTEAD
    Calculate the area under the ROC curve for a given predictor and horizon
    Higher is better. Perfect score is 1, random score is 0.5
    
    Parameters
    ----------
    predictor : DisruptionPredictor
        The predictor to evaluate
    horizon : float
        How far into the future the predictor is looking
    data : pandas.DataFrame
        The data to evaluate the predictor on
        Sorted by shot number and time
        Expected to already be transformed by the predictor's transformer
    """
    
    # Set the thresholds to use
    thresholds = np.linspace(0, 1, 100)

    # Create arrays to store the results
    # Array is of shape (num_shots, num_thresholds)
    true_positives = np.zeros((len(data), len(thresholds)))
    false_positives = np.zeros((len(data), len(thresholds)))

    # Get a running total of number of disruptive shots
    num_disruptive = 0

    # Iterate through shots
    for i, entry in enumerate(data):
        disrupt = entry[0]
        shot_data = entry[1]
        # Calculate the disruption time
        disruption_times = predictor.calculate_disruption_time(shot_data, thresholds, horizon)

        # Fill in true positives and false positives
        true_positives[i] = np.array([disrupt and (disruption_time is not None) for disruption_time in disruption_times])
        false_positives[i] = np.array([(not disrupt) and (disruption_time is not None) for disruption_time in disruption_times])

        if disrupt:
            num_disruptive += 1

    # Calculate the true positive rate and false positive rate for each threshold
    true_positive_rates = np.sum(true_positives, axis=0) / num_disruptive
    false_positive_rates = np.sum(false_positives, axis=0) / (len(data) - num_disruptive)


    # Calculate the area under the ROC curve
    # Use the trapezoidal rule
    au_roc = -np.trapz(true_positive_rates, false_positive_rates)

    return au_roc

def calc_tp_fp_times(predictor:DisruptionPredictor, horizon, data, thresholds):
    """
    Calculate the True Positives, False Positives, and Warning Times
    at a given horizon for a given predictor and thresholds.
    """

    # Create arrays to store the results
    # Array is of shape (num_shots, num_thresholds)
    true_positives = np.zeros((len(data), len(thresholds)))
    false_positives = np.zeros((len(data), len(thresholds)))

    # Create list to store warning times
    # This is a list of arrays of variable length,
    # but the arrays will line up such that each index corresponds to the same threshold
    warning_times = []

    # Iterate through shots
    for i, entry in enumerate(data):
        disrupt = entry[0]
        shot_data = entry[1]
        # Calculate the disruption time predicted by the model
        predicted_times = predictor.calculate_disruption_time(shot_data, thresholds, horizon)

        # Fill in true and false positives
        true_positives[i] = np.array([disrupt and (predicted_time is not None) for predicted_time in predicted_times])
        false_positives[i] = np.array([(not disrupt) and (predicted_time is not None) for predicted_time in predicted_times])

        # If shot is disruptive, can fill in Time to First True Detection
        if disrupt:
            # Find actual disruption time by looking at last time in shot
            true_time = shot_data['time'].iloc[-1]

            warning_times.append(np.array([true_time - predicted_time for predicted_time in predicted_times if predicted_time is not None]))

    return true_positives, false_positives, warning_times

def calc_num_disrupt(data):
    """
    Find number of disruptive shots in data
    """

    num_disruptive = 0

    # Iterate through shots
    for entry in data:
        disrupt = entry[0]
        if disrupt:
            num_disruptive += 1

    return num_disruptive

def benchmark_warning_time(predictor:DisruptionPredictor, horizon, device, dataset):
    """
    Calculate the warning time curve at a single time horizon.
    This is the average and standard deviation of the warning time
    for a given predictor vs false positive rate,
    where warning time is defined as the time between predicted onset and the disruption actually happening.
    Used to compare performance of different predictors like in Fig. 3 of 
    Dynamically Personalized Detection of Hemorrhage, Chirag et al 2019.
    (NOTE: these are diferent metrics, but the plots look similar and they convey similar information)

    Parameters
    ----------
    predictor : DisruptionPredictor
        The predictor to evaluate
    horizon : float
        How far into the future the predictor is looking
    device : str
        The device to evaluate the predictor on
    dataset : str
        The dataset to evaluate the predictor on
        Should be a similar dataset to the one the predictor was trained on
        and that the predictor's transformer was calculated for
    Returns:
    --------
    false_positive_rates : list of float
        The false positive rates
    mean_warning_times : list of float
        The average warning time for each false positive rate
    std_warning_times : list of float
        The standard deviation of the warning time for each false positive rate
    """

    # Load the data and process it with predictor's transformer
    data = load_benchmark_data(predictor, device, dataset)

    # Set the thresholds to use
    thresholds = np.linspace(0, 1, 1000)

    # Calculate the false positives, and warning times
    _, false_positives, warning_times = calc_tp_fp_times(predictor, horizon, data, thresholds)

    num_disruptive = calc_num_disrupt(data)

    # The way this is formatted at this point isn't really 'warning times' vs 'FPR' but rather
    # 'warning times at threshold' and 'FPR at threshold'
    # When returned, the warning time statistics will correspond to a particular FPR

    # Calculate the false positive rate for each threshold
    false_positive_rates = np.sum(false_positives, axis=0) / (len(data) - num_disruptive)

    mean_warning_times = []
    std_warning_times = []

    # TODO: Should really really vectorize this

    # Calculate the average warning time for each false positive rate
    fpr_times = []
    for i in range(len(thresholds)):
    
        for warning_time in warning_times:
            try:
                fpr_times.append(warning_time[i])
            except IndexError:
                # This is a disruptive shot that didn't have a detection at this threshold
                # Warning time is 0
                fpr_times.append(0)
        
        # Clump the detection times that share a false positive rate together
        # Or if we're at the end, we need to add the last one regardless
        if i == len(thresholds) - 1 or (false_positive_rates[i] != false_positive_rates[i+1]):
            if len(fpr_times) > 0:
                mean_warning_times.append(np.mean(fpr_times))
                std_warning_times.append(np.std(fpr_times))
                fpr_times = []
            else:
                # If there are no detection times, that means false positive rate is 0. Detection time is 0.
                mean_warning_times.append(0)
                std_warning_times.append(0)
            

    # Eliminate duplicate false positive rates.
    # However, this sorts the false positive rates, so we need to reverse the order afterwards
    unique_false_positive_rates = np.unique(false_positive_rates)
    # Reverse the order so that the false positive rates are increasing (to once again line up with the detection times)
    unique_false_positive_rates = unique_false_positive_rates[::-1]

    # Ignore zero false positve rate results
    return unique_false_positive_rates[:-1], mean_warning_times[:-1], std_warning_times[:-1]