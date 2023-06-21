import io
import pkgutil

import numpy as np
import pandas as pd


DEFAULT_FEATURES = ['ip','Wmhd','n_e','kappa','li']

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
        In the form of data sorted first by shot then by time
    """
    data = pkgutil.get_data(__name__, 'data/{}/{}.csv'.format(device, dataset))
    data = pd.read_csv(io.BytesIO(data)) # type: ignore
    # Sort by shot number and time
    data = data.sort_values(['shot','time'])
    return data

def parse_dataset(device, dataset, epsilon=1e-4):
    """ Parse the dataset from the given device
    adds a 'time_to_last' column to the dataset
    Replace zeroes in time_until_disruption with epsilon
    
    """
    
    data = load_dataset(device,dataset)

    # Create a 'time to last measurement' column for each shot
    # This is the time from the present time to the last measurement
    data['time_to_last'] = data.groupby('shot')['time'].transform(max) - data['time']

    # Replace zeros in times with low-value epsilon
    data['time_to_last'] = data['time_to_last'].replace(0, epsilon)
    data['time_until_disrupt'] = data['time_until_disrupt'].replace(0, epsilon)

    return data

def load_features_outcomes(device, dataset, features=None):
    """ Load the specified dataset from a device,
    and return the features and outcomes for usage in SurvivalModel
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
    
    data = parse_dataset(device, dataset)

    outcomes = pd.DataFrame()
    
    # Outcomes 'event' is 1 if time_until_disrupt is not null
    outcomes['event'] = data['time_until_disrupt'].notnull().astype(int)

    # Outcomes 'time' is time_until_disrupt if it is not null, and time_to_last otherwise
    outcomes['time'] = data['time_until_disrupt'].fillna(data['time_to_last'])

    # make outcomes is the correct type
    outcomes = outcomes.astype({'event': 'int64', 'time': 'float64'})

    # trim data to only include features used for training
    if features is None:
        features = DEFAULT_FEATURES

    return data[features], outcomes


def load_features_events_indicators(device, dataset, features=None):
    """ For usage in Recurrent DSM and Recurrent CPH
    Load the specified dataset from a device,
    and return the features and outcomes and event indicators,
    grouped by shot number

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
    """

    data = parse_dataset(device, dataset)

    # TODO fill this out after get DPRF working
    return None



def load_features_labels(device, dataset, cutoff_threshold, features=None):
    """For usage in Random Forest Classifier
    
    """

    data = parse_dataset(device, dataset)

    # label is '1' if time_until_disrupt is less than cutoff_threshold, '0' otherwise
    labels = (data['time_until_disrupt'] < cutoff_threshold).astype(int)

    # trim data to only include features used for training
    if features is None:
        features = DEFAULT_FEATURES
    
    return data[features], labels

def make_training_sets(device, dataset, random_seed=0):
    """
    Split the raw data into training, test, and validation sets
    """

    # Load the raw dataset
    data = load_dataset(device, dataset+ '_raw')

    # Eliminate timeslices with null values in any feature except time_until_disrupt
    data = data.dropna(subset=['ip', 'Wmhd', 'n_e', 'kappa', 'li'])
    # Eliminate timeslices with negative values in time
    data = data[data['time'] >= 0]

    # Remove where time_until_disrupt is negative, keeping where time_until_disrupt is null
    data = data[(data['time_until_disrupt'] >= 0) | (data['time_until_disrupt'].isnull())]

    # Find the unique shots in the dataset
    shots = data['shot'].unique()

    # Shuffle the shots
    np.random.seed(random_seed)
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

def get_disruptive_shot_list(device, dataset):
    """
    Get the list of disruptive shots from the dataset
    """

    # Load the raw dataset
    data = load_dataset(device, dataset)

    # Find the disruptive shots in the dataset
    disruptive_shots = data[data['time_until_disrupt'] >= 0]['shot'].unique()

    # Return the list of disruptive shots
    return disruptive_shots

def get_non_disruptive_shot_list(device, dataset):
    """
    Get the list of non-disruptive shots from the dataset
    """

    # Load the raw dataset
    data = load_dataset(device, dataset)

    # Find the non-disruptive shots in the dataset
    non_disruptive_shots = data[data['time_until_disrupt'].isnull()]['shot'].unique()

    # Return the list of non-disruptive shots
    return non_disruptive_shots

def load_dataset_grouped(device, dataset):
    """
    Load dataset grouped by shot number
    """

    # Load the raw dataset
    data = load_dataset(device, dataset)

    # Group the data by shot number
    group_data = data.groupby('shot')

    # Return the grouped dataset
    return group_data