"""Utilities to evaluate the performance of a disruption predictor"""

from DisruptionPredictor import DisruptionPredictor

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

    # Load the data

    # Transform the data

    # Iterate through horizons

    # Return

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
    
    # Iterate through shots

    # Set the thresholds

    # Calculate the disruption time for each shot

    # Find the true positive and false positive rates

    # Find the ROC curve

    # Calculate the area under the ROC curve


