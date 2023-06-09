import io
import pkgutil

import numpy as np
import pandas as pd


def load_dataset(device, dataset):
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


def load_features_outcomes(device, dataset):
    """ Load the specified dataset from a device,
    and return the features and outcomes for usage in survival analysis
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
    data = load_dataset(device, dataset)

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
    features = ['ip','Wmhd','n_e','kappa','li']

    return outcomes, data[features]


def make_training_sets(device, dataset):
    """
    Split the raw data into training, test, and validation sets
    """

    # Load the raw dataset
    data = load_dataset(device, dataset+ '_raw')

    # Eliminate timeslices with null values in any feature except time_until_disrupt
    data = data.dropna(subset=['ip', 'Wmhd', 'n_e', 'kappa', 'li'])

    # Find the unique shots in the dataset
    shots = data['shot'].unique()

    # Shuffle the shots
    np.random.shuffle(shots)

    # Split the shots into training, test, and validation sets
    train_shots = shots[:int(len(shots)*0.6)]
    test_shots = shots[int(len(shots)*0.6):int(len(shots)*0.8)]
    val_shots = shots[int(len(shots)*0.8):]

    # Split the data into training, test, and validation sets
    train_data = data[data['shot'].isin(train_shots)]
    test_data = data[data['shot'].isin(test_shots)]
    val_data = data[data['shot'].isin(val_shots)]

    # Save the training, test, and validation sets
    train_data.to_csv('data/{}/{}_train.csv'.format(device, dataset), index=False)
    test_data.to_csv('data/{}/{}_test.csv'.format(device, dataset), index=False)
    val_data.to_csv('data/{}/{}_val.csv'.format(device, dataset), index=False)

    # Print the number of shots in each set
    print('Training shots: {}'.format(len(train_shots)))
    print('Test shots: {}'.format(len(test_shots)))
    print('Validation shots: {}'.format(len(val_shots)))
