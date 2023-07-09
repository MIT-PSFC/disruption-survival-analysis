"""Utilities to evaluate the performance of a disruption predictor"""

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from preprocess_datasets import load_dataset_grouped, get_disruptive_shot_list
from DisruptionPredictors import DisruptionPredictor

def load_benchmark_data(predictor:DisruptionPredictor, device, dataset):
    
    # Get a list of all disruptive shots (disruption happens during flattop)
    disruptive_shots = get_disruptive_shot_list(device, dataset)

    # Load the data grouped by shot number
    raw_data = load_dataset_grouped(device, dataset)
    
    data = []
    for entry in raw_data:
        # Replace the shot numbers with a boolean indicating if the shot is disruptive
        shotnumber = entry[0]
        disrupt = shotnumber in disruptive_shots

        # Trim the raw data to only include the features used by the predictor
        # and apply the transformer
        raw_shot_data = entry[1]
        shot_data = predictor.transformer.transform(raw_shot_data[predictor.features])
        # Put the times back in
        shot_data['time'] = raw_shot_data['time']
        data.append((disrupt, shot_data))

    return data

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
        # Might need to use 'weighted' option in roc_auc_score to account for this
        # Doesn't appear to make much of a difference.
        # Calculate the area under the ROC curve 
        au_roc = roc_auc_score(y_true, y_score)

        #au_roc = calc_au_roc(predictor, horizon, data)
        au_rocs.append(au_roc)

    return au_rocs

def label_shot_data(shot_data, disrupt, horizon):
    """
    Label the data as disruptive or not disruptive at a given horizon
    Parameters
    ----------
    data : pandas.DataFrame
        The data to label
    disrupt : bool
        If the shot is disruptive
    horizon : float
        How far into the future to look
    Returns
    -------
    labeled_data : numpy.ndarray
        An array of booleans indicating if the time slice is disruptive or not
    """

    if disrupt:
        # If the shot is disruptive, label all time slices up to
        # horizon seconds before the disruption as non-disruptive
        # and all time slices after horizon seconds before the disruption as disruptive
        # Labels are either 0 (non-disruptive) or 1 (disruptive)
        disruption_time = shot_data['time'].max()
        labeled_data = np.array(shot_data['time'] < (disruption_time - horizon)).astype(int)
    else:
        # If the shot is not disruptive, label all time slices as non-disruptive
        labeled_data = np.zeros(len(shot_data))

    return labeled_data

def calc_au_roc(predictor:DisruptionPredictor, horizon, data):
    """
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

def benchmark_true_detection(predictor:DisruptionPredictor, horizon, device, dataset):
    """
    Calculate the Activity Monitoring Operator Characteristic (AMOC) curve at a single time horizon.
    This is the average and standard deviation of the Time to First True Detection (T2FD)
    for a given predictor vs false positive rate.
    Used to compare performance of different predictors like in Fig. 3 of 
    Dynamically Personalized Detection of Hemorrhage, Chirag et al 2019.

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
    mean_detection_times : list of float
        The average detection time for each false positive rate
    std_detection_times : list of float
        The standard deviation of the detection time for each false positive rate
    """

    # Load the data and process it with predictor's transformer
    data = load_benchmark_data(predictor, device, dataset)

    # Set the thresholds to use
    thresholds = np.linspace(0, 1, 100)

    # Create arrays to store the results
    # Array is of shape (num_shots, num_thresholds)
    false_positives = np.zeros((len(data), len(thresholds)))

    # Create list to store detection times for each shot
    # This is a list of arrays of variable length,
    # but the arrays will line up such that each index corresponds to the same threshold
    detection_times = []

    # Get a running total of number of disruptive shots
    num_disruptive = 0

    # Iterate through shots
    for i, entry in enumerate(data):
        disrupt = entry[0]
        shot_data = entry[1]
        # Calculate the disruption time predicted by the model
        disruption_times = predictor.calculate_disruption_time(shot_data, thresholds, horizon)

        # Fill in false positives
        false_positives[i] = np.array([(not disrupt) and (disruption_time is not None) for disruption_time in disruption_times])

        if disrupt:
            num_disruptive += 1
            
            # If shot is disruptive, can fill in Time to First True Detection

            # Find actual disruption time by looking at last time in shot
            onset_time = shot_data['time'].iloc[-1]

            detection_times.append(np.array([onset_time - disruption_time for disruption_time in disruption_times if disruption_time is not None]))

    # The way this is formatted at this point isn't really 'T2FD' vs 'FPR' but rather
    # 'T2FD at threshold' and 'FPR at threshold'
    # When returned, the average T2FDs will correspond to a particular FPR

    # Calculate the false positive rate for each threshold
    false_positive_rates = np.sum(false_positives, axis=0) / (len(data) - num_disruptive)

    mean_detection_times = []
    std_detection_times = []

    # TODO: Should really really vectorize this

    # Calculate the average detection time for each false positive rate
    threshold_times = []
    for i in range(len(thresholds)):
    
        for detection_time in detection_times:
            try:
                threshold_times.append(detection_time[i])
            except IndexError:
                pass
        
        # Clump the detection times that share a false positive rate together
        if i != 0 and false_positive_rates[i] != false_positive_rates[i-1]:
            mean_detection_times.append(np.mean(threshold_times))
            std_detection_times.append(np.std(threshold_times))
            threshold_times = []
        elif i == len(thresholds) - 1:
            # If we're at the end, we need to add the last one
            mean_detection_times.append(np.mean(threshold_times))
            std_detection_times.append(np.std(threshold_times))

    # Eliminate duplicate false positive rates
    unique_false_positive_rates = np.unique(false_positive_rates)

    return unique_false_positive_rates, mean_detection_times, std_detection_times