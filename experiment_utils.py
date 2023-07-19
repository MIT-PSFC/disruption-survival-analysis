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
