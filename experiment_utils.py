# Functions used by experiments but not actually part of the experiments

import numpy as np

from Experiments import Experiment

from DisruptionPredictors import DisruptionPredictorSM, DisruptionPredictorRF, DisruptionPredictorKM

from auton_survival.estimators import SurvivalModel # CPH, DCPH, DSM, DCM, RSF
from sklearn.ensemble import RandomForestClassifier

from model_utils import get_model_for_experiment, name_model
from manage_datasets import load_dataset


SURVIVAL_MODELS = ['cph', 'dcph', 'dcm', 'dsm']
BINARY_CLASSIFIERS = ['rf', 'km']

def make_experiment(config, experiment_type):
    """
    Make an experiment from a config dictionary. 
    If the experiment type is 'test', then the experiment will be a test experiment.
    If the experiment type is 'val', then the experiment will be a validation experiment.

    Parameters
    ----------
    config : dict
        Dictionary of everything unique to this experiment.
        Should contain the model type, the metric to be evaluated, which dataset to use, and some model-specific hyperparameters
    experiment_type : str
        The type of experiment to make. Either 'test' or 'val'
        
    Returns
    -------
    experiment : Experiment
        The experiment to be run

    """

    # Create the model and predictor for the experiment
    model = get_model_for_experiment(config, experiment_type)

    required_warning_time = config['01_required_warning_time']

    name = name_model(config)

    if isinstance(model, SurvivalModel):
        predictor = DisruptionPredictorSM(name, model, required_warning_time, config['horizon'])
    elif isinstance(model, RandomForestClassifier):
        predictor = DisruptionPredictorRF(name, model, required_warning_time, config['class_time'])
    else:
        raise ValueError('Model type not recognized')
    
    # Load data for the experiment
    all_data = load_dataset(config['00_device'], config['00_dataset_path'], experiment_type)

    experiment = Experiment(name, all_data, predictor, experiment_type, config['01_alarm_type'])

    return experiment

def label_shot_data(shot_data, disrupt, disruptive_window):
    """
    Label the data as disruptive or not disruptive based on the disruptive window
    Parameters
    ----------
    data : pandas.DataFrame
        The data to label
    disrupt : bool
        If the shot is disruptive
    disruptive_window : float
        Time before a disruption to label shot data as disruptive (in seconds)
    Returns
    -------
    labeled_data : numpy.ndarray
        An array of booleans indicating if the time slice is disruptive or not
    """

    if disrupt:
        # If the shot disrupts, label all time slices up to
        # disruptive_window seconds before the disruption as non-disruptive
        # and all time slices after disruptive_window seconds before the disruption as disruptive
        # Labels are either 0 (non-disruptive) or 1 (disruptive)
        disruption_time = shot_data['time'].max()
        labeled_data = np.array(shot_data['time'] > (disruption_time - disruptive_window)).astype(int)
    else:
        # If the shot is not disruptive, label all time slices as non-disruptive
        labeled_data = np.zeros(len(shot_data))

    return labeled_data

def calculate_alarm_times(risk_at_time, thresholds):
    """
    Calculates the alarm times for a given shot with a simple threshold

    Parameters
    ----------
    risk_at_time : pandas.DataFrame
        The risk of disruption for each time slice in a single shot
        Should be sorted by time
        Should be transformed by the predictor's transformer
    thresholds : list of float
        The thresholds to use for determining if a disruption is imminent
        Expects a list of floats between 0 and 1, sorted from lowest to highest
        Disruption is predicted when the risk exceeds a threshold

    Returns
    -------
    alarm_times : list of float
        The times of alarm (predicted disruption)
        If no disruption is predicted, returns None in that position
    """
    
    # Make a copy of the thresholds it to keep track of which have been used
    if isinstance(thresholds, np.ndarray):
        avail_thresholds = thresholds.tolist()
    else:
        avail_thresholds = thresholds.copy()

    alarm_times = []
    # Go through the shot data and find the first time the risk exceeds each threshold
    for i in range(len(risk_at_time)):
        # If there are no more thresholds, stop
        if len(avail_thresholds) == 0:
            break

        # If the risk ever exceeds the threshold, add the time to the list
        # and remove the threshold from the list
        # Then keep going until the risk is below the next threshold
        # or there are no more thresholds
        risk = risk_at_time.iloc[i]['risk']
        while risk > avail_thresholds[0]:
            alarm_times.append(risk_at_time.iloc[i]['time'])
            avail_thresholds.pop(0)
            if len(avail_thresholds) == 0:
                break
    
    # If there is a mismatch between alarm times and thresholds,
    # fill in the rest of the alarm times with None
    if len(alarm_times) < len(thresholds):
        for i in range(len(thresholds) - len(alarm_times)):
            alarm_times.append(None)

    # Return the alarm times
    return alarm_times

def calculate_alarm_times_hysteresis(risk_at_time, thresholds):
    """
    Calculates the alarm times for a given shot with hysterisis method
    If the 'disruptivity' output of the model goes above the upper threshold
    and remains above the lower threshold for the window length, a disruption
    is predicted

    Parameters
    ----------
    risk_at_time : pandas.DataFrame
        The risk of disruption for each time slice in a single shot
        Should be sorted by time
        Should be transformed by the predictor's transformer
    thresholds : list of tuple of (float, float, float)
        The thresholds to use for determining if a disruption is imminent
        Expects a list of tuples of the form (lower_threshold, upper_threshold, window_length)
        Disruption is predicted when the risk exceeds the upper threshold
        and remains above the lower threshold for the window length (Same implementation as ENI script, maybe same as CERN?)

    """
    raise NotImplementedError("Hysterisis method not yet implemented")

def calculate_alarm_times_ettd(ettd_at_time, thresholds):
    """
    Calculates the alarm times for a given shot using the expected time to disruption.
    If the expected time to disruption output of the model drops below some threshold,
    a disruption is predicted.
    """

    raise NotImplementedError("Expected Time To Disruption method not yet implemented")

def clump_many_to_one_statistics(unique_values_raw, clumping_values, epsilon=0.01):
    """
    For example, The way this is calculated, the warning times and true alarm rates and false alarm rates are all given by particular thresholds
    As such, we can easily compare them to the thresholds, since each value corresponds to one threshold
    However, when comparing them to eachother, this becomes difficult because there is not necessarily a one-to-one correspondence
    For instance, we could have a true alarm rate of 0.5 for both a threshold of 0.1 and 0.2, but the warning times could be different
    This function clumps values together for each unique value

    Parameters
    ----------
    unique_values_raw : numpy.ndarray
        The unique values which dictate the clumping. Not necessarily unique at this point.
    clumping_values : numpy.ndarray
        The values to be clumped. Must be the same length as unique_values.
    epsilon : float
        The maximum difference between two unique values to be considered the same

    Returns
    -------
    unique_true_alarm_rates : numpy.ndarray
        The unique true alarm rates
    avg_warning_times : numpy.ndarray
        The average clumped value for each unique value
    std_warning_times : numpy.ndarray
        The standard deviation of the clumped values for each unique value
    
    """

    # Within these unique values, if they are within epsilon of eachother, eliminate one
    # TODO

    # Actually trim down to the unique values
    unique_values = np.unique(unique_values_raw)

    # Initialize the average and standard deviation arrays
    avg_clump_values = np.zeros(len(unique_values))
    std_clump_values = np.zeros(len(unique_values))

    # Go through each unique value and find the clumping values which correspond to it
    for i, unique_value in enumerate(unique_values):
        # Find the indices of the raw unqiue values that correspond to this particular unique value
        indices = np.where(unique_values_raw == unique_value)

        # Get the clumping values that correspond to this unique value
        # TODO: polish up this list comprehension to be more readable
        try:
            clumping_values_2D = [clumping_values[k][j] for k in range(len(clumping_values)) for j in indices]
            # Flatten the list
            clumping_values_1D = [item for sublist in clumping_values_2D for item in sublist]
        except:
            clumping_values_1D = [clumping_values[j] for j in indices]

        # Calculate the average and standard deviation of the clumped values
        avg_clump_values[i] = np.mean(clumping_values_1D)
        std_clump_values[i] = np.std(clumping_values_1D)

    return unique_values, avg_clump_values, std_clump_values
