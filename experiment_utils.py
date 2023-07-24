# Functions used by experiments but not actually part of the experiments

import numpy as np

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
        # If the shot disrupts, label all time slices up to
        # horizon seconds before the disruption as non-disruptive
        # and all time slices after horizon seconds before the disruption as disruptive
        # Labels are either 0 (non-disruptive) or 1 (disruptive)
        disruption_time = shot_data['time'].max()
        labeled_data = np.array(shot_data['time'] > (disruption_time - horizon)).astype(int)
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

def calculate_alarm_time_hysterisis(self, shot_data, 
                                            upper_threshold, lower_threshold, 
                                            window, horizons):
    """
    Calculates the alarm times for a given shot with hysterisis method
    If the 'disruptivity' output of the model goes above the upper threshold
    and remains above the lower threshold for the window length, a disruption
    is predicted
    """
    raise NotImplementedError("Hysterisis method not yet implemented")

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
