"""Utilities to evaluate the performance of a disruption predictor"""

import numpy as np
import pandas as pd

from preprocess_datasets import load_dataset_grouped, get_disruptive_shot_list
from DisruptionPredictors import DisruptionPredictor

def benchmark(predictor:DisruptionPredictor, horizons, device, dataset):
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

    # Load the data grouped by shot number
    raw_data = load_dataset_grouped(device, dataset)

    # Get a list of all disruptive shots (disruption happens during flattop)
    disruptive_shots = get_disruptive_shot_list(device, dataset)

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

    # TODO: calculating all horizons at once would be more efficient, but not working yet
    #au_rocs = calc_au_roc(predictor, horizons, data)

    # Calculate area under the ROC curve for all horizons
    au_rocs = []
    for horizon in horizons:
        # Calculate the area under the ROC curve
        au_roc = calc_au_roc(predictor, horizon, data)
        au_rocs.append(au_roc)

    return au_rocs

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
    thresholds = np.linspace(0, 1, 10)

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
        false_positives[i] = np.array([not disrupt and (disruption_time is not None) for disruption_time in disruption_times])

        if disrupt:
            num_disruptive += 1

    # Calculate the true positive rate and false positive rate for each threshold
    true_positive_rates = np.sum(true_positives, axis=0) / num_disruptive
    false_positive_rates = np.sum(false_positives, axis=0) / (len(data) - num_disruptive)

    # Find the ROC curve
    # Sort by false positive rate
    sort_indices = np.argsort(false_positive_rates)
    true_positive_rates_roc = true_positive_rates[sort_indices]
    false_positive_rates_roc = false_positive_rates[sort_indices]

    # Calculate the area under the ROC curve
    # Use the trapezoidal rule
    au_roc = np.trapz(true_positive_rates_roc, false_positive_rates_roc)

    return au_roc


