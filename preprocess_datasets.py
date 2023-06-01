import io
import pkgutil

import pandas as pd


def load_dataset(device='cmod', dataset='small_set'):
    """ Load the dataset from the given device and dataset
    Parameters
    ----------
    device : str
        The device for which to load the data
    dataset : str
        The dataset to load
    Returns
    -------
    data : pandas.DataFrame
        A dataframe containing the data in the dataset
        In the form of unsorted timeslices of all included shots
    """
    data = pkgutil.get_data(__name__, 'data/{}/{}.csv'.format(device, dataset))
    data = pd.read_csv(io.BytesIO(data))
    return data


def load_dataset_train(device='cmod', dataset='small_set'):
    """ Load the specified dataset from a device,
    and return the features and outcomes for training
    Parameters
    ----------
    device : str
        The device for which to load the data
    dataset : str
        The dataset to load
    Returns
    -------
    outcomes : pandas.DataFrame
        A dataframe containing the outcomes for each shot
        Outcomes are whether or not the event occurred, and the time until the event or last measurement
    features : pandas.DataFrame
        A dataframe containing the features for each shot
        TODO: should this match DPRF ordering?
    """
    data = load_dataset(device=device, dataset=dataset)

    # Add time of last measurement for each shot
    data['last_time'] = data.groupby('shot')['time'].transform('max')

    # For each timeslice, determine if the shot was disrupted
    # And the time until disruption or last measurement
    outcomes = data.copy()
    # Outcomes 'event' is 1 if time_until_disrupt is not null
    outcomes['event'] = outcomes['time_until_disrupt'].notnull().astype(int)
    
    outcomes['time'] = data['time_until_disrupt'].fillna(data['last_time'] - data['time'])
    outcomes = outcomes[['event', 'time']]

    
    # trim data to only include features used for training
    # TODO: add in aminor, squareness, and triangularity 
    train_feats = ['ip','Wmhd','n_e','kappa','li']

    return outcomes, data[train_feats]

