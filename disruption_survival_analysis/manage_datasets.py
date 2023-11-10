import io
import os
import pkgutil

import numpy as np
import pandas as pd

from auton_survival.preprocessing import Preprocessor

NOT_FEATURES = ['time', 'shot', 'time_until_disrupt'] # Columns that are not features
DROPPED_FEATURES = ['v_surf', 'dbetap_dt', 'dli_dt', 'dWmhd_dt', 'dn_dt', 'dip_dt', 'dip_smoothed', 'dipprog_dt', 'ip_prog'] # Additional features to drop from the raw dataset (should not wind up in the training datasets)

DISRUPTION_DROPPED_TIME = 0.000 # Time before disruption to drop from the dataset (to avoid noise from disruption)

# Functions for making new datasets

def make_training_sets(device, dataset_path, random_seed=0, debug=False):
    """
    Split the raw data into training, test, and validation sets,
    where all data is transformed based on the training set.
    Expects there to be a 'raw.csv' dataset in a data folder in the current or an above directory.

    Parameters
    ----------
    device : str
        The device to use
    dataset_path : str
        The path to the dataset
    random_seed : int
        The random seed to use for shuffling the shots
    debug : bool
        Whether to print debug statements (how many shots are in each dataset)

    Returns
    -------
    None
    """

    # Load the raw dataset
    data = load_dataset(device, dataset_path, 'raw')

    # Eliminate dropped features if they exist
    for col in DROPPED_FEATURES:
        if col in data.columns:
            data = data.drop(columns=[col])
    # Eliminate timeslices with null values in any feature except time_until_disrupt
    data = data.dropna(subset=[col for col in data.columns if col not in ['time_until_disrupt']])
    # Eliminate timeslices with negative values in time
    data = data[data['time'] >= 0]
    # Remove where time_until_disrupt is negative, keeping where time_until_disrupt is null
    data = data[(data['time_until_disrupt'] >= DISRUPTION_DROPPED_TIME) | (data['time_until_disrupt'].isnull())]
    
    # Find the shots that have a disruption
    disrupt_shots = load_disruptive_shot_list(device, dataset_path, 'raw')
    # Find the shots that do not have a disruption
    non_disrupt_shots = load_non_disruptive_shot_list(device, dataset_path, 'raw')

    np.random.seed(random_seed)
    # Shuffle the disruptive and non-disruptive shots
    np.random.shuffle(disrupt_shots)
    np.random.shuffle(non_disrupt_shots)

    # Split each set of shots into training, test, and validation sets
    disrupt_train_shots = disrupt_shots[:int(len(disrupt_shots)*0.6)]
    disrupt_test_shots = disrupt_shots[int(len(disrupt_shots)*0.6):int(len(disrupt_shots)*0.8)]
    disrupt_val_shots = disrupt_shots[int(len(disrupt_shots)*0.8):]

    non_disrupt_train_shots = non_disrupt_shots[:int(len(non_disrupt_shots)*0.6)]
    non_disrupt_test_shots = non_disrupt_shots[int(len(non_disrupt_shots)*0.6):int(len(non_disrupt_shots)*0.8)]
    non_disrupt_val_shots = non_disrupt_shots[int(len(non_disrupt_shots)*0.8):]

    # Split the data into training, test, and validation sets
    train_data = data[data['shot'].isin(disrupt_train_shots) | data['shot'].isin(non_disrupt_train_shots)]
    test_data = data[data['shot'].isin(disrupt_test_shots) | data['shot'].isin(non_disrupt_test_shots)]
    val_data = data[data['shot'].isin(disrupt_val_shots) | data['shot'].isin(non_disrupt_val_shots)]

    # Get the features
    features = [col for col in data.columns if col not in ['shot', 'time_until_disrupt', 'time']]

    # Transform the data based on the training set
    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
    transformer = preprocessor.fit(train_data[features], cat_feats=[], num_feats=features, one_hot=True, fill_value=-1)

    # Transform the training, test, and validation sets
    train_data[features] = transformer.transform(train_data[features])
    test_data[features] = transformer.transform(test_data[features])
    val_data[features] = transformer.transform(val_data[features])

    # Save the training, test, and validation sets
    try:
        train_data.to_csv('../data/{}/{}/train_full.csv'.format(device, dataset_path), index=False)
        test_data.to_csv('../data/{}/{}/test.csv'.format(device, dataset_path), index=False)
        val_data.to_csv('../data/{}/{}/val.csv'.format(device, dataset_path), index=False)
    except:
        train_data.to_csv('data/{}/{}/train_full.csv'.format(device, dataset_path), index=False)
        test_data.to_csv('data/{}/{}/test.csv'.format(device, dataset_path), index=False)
        val_data.to_csv('data/{}/{}/val.csv'.format(device, dataset_path), index=False)

    # Print the number of shots in each set
    if debug:
        print('Training shots: {}'.format(len(disrupt_train_shots)+len(non_disrupt_train_shots)))
        print('Test shots: {}'.format(len(disrupt_test_shots)+len(non_disrupt_test_shots)))
        print('Validation shots: {}'.format(len(disrupt_val_shots)+len(non_disrupt_val_shots)))

def make_stacked_sets(device, dataset_path, dataset_category, stack_size):
    """Given some dataset, stack the features so each timeslice includes itself and the stack_size previous timeslices
    Fills with 0's if there are not enough previous timeslices
    NOTE: Only use this if the sampling rate of the dataset is constant

    Parameters
    ----------
    device : str
        The device to use
    dataset_path : str
        The path to the dataset
    dataset_category : str
        The category of the dataset. Should be one of 'train', 'test', or 'val'
    stack_size : int
        The number of timeslices to stack into a single feature.

    Returns
    -------
    None
    """

    data = load_dataset(device, dataset_path, dataset_category)

    # Get the columns to stack
    feature_list = load_feature_list(device, dataset_path)

    extended_feature_list = feature_list.copy()
    for feature in feature_list:
        for i in range(stack_size):
            extended_feature_list.append(f'{feature}_{i}')

    stacked_features = pd.DataFrame(columns=extended_feature_list)

    # Get the shot numbers
    shot_numbers = data['shot'].unique()

    # For each shot, stack the features

    for shot in shot_numbers:

        shot_stack = data[data['shot'] == shot].copy()

        for feature in feature_list:
            for i in range(stack_size):
                shot_stack[f'{feature}_{i}'] = shot_stack[feature].shift(stack_size)

        # Replace all feature NaN's with 0's
        shot_stack[extended_feature_list] = shot_stack[extended_feature_list].fillna(0)

        stacked_features = pd.concat([stacked_features, shot_stack], ignore_index=True)
    
    # Save stacked features under new name
    new_dataset_path = f'data/{device}/{dataset_path}/stack_{stack_size}'

    # If directory does not exist, create it
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    stacked_features.to_csv(new_dataset_path + f'/{dataset_category}.csv', index=False)

def focus_training_set(device, dataset_path, random_seed=0):
    """ Take the training set and remove some data from the non-disruptive shots to improve the class balance
    in the training set.
    
    Parameters
    ----------
    device : str
        The device to use
    dataset_path : str
        The path to the dataset
    random_seed : int, optional
        The random seed to use for picking which data to remove
    """

    # Load the training set
    data = load_dataset(device, dataset_path, 'train_full')

    # Find the list of non-disruptive shots
    non_disrupt_shots = load_non_disruptive_shot_list(device, dataset_path, 'train_full')

    # Make a new training set with the same columns
    new_data = pd.DataFrame(columns=data.columns)

    # Iterate through each disruptive shot
    for shot in non_disrupt_shots:
            
        # Get the timeslices for the shot
        shot_data = data[data['shot'] == shot]

        # Find the number of timeslices in the shot
        num_timeslices = len(shot_data)

        # Find the number of timeslices to remove
        num_remove = int(num_timeslices * 0.9)

        # Choose the timeslices to remove
        remove_indices = np.random.choice(num_timeslices, size=num_remove, replace=False)

        # Remove the chosen timeslices
        shot_data = shot_data.drop(shot_data.index[remove_indices])

        # Add the remaining timeslices to the new training set
        new_data = pd.concat([new_data, shot_data], ignore_index=True)

        # Remove data from the shot in the original training set
        data = data[data['shot'] != shot]

    # Add the disruptive shots to the new training set
    new_data = pd.concat([new_data, data], ignore_index=True)
    
    # Save the new training set
    new_data.to_csv(f'data/{device}/{dataset_path}/train.csv', index=False)

# Functions for loading datasets or other information directly from saved .csv files

def load_dataset(device, dataset_path, dataset_category):
    """ Load a dataset from a specific device, sorted by shot number and time. Does not do any further processing.

    Parameters
    ----------
    device : str
        The device for which to load the data
    dataset_path : str
        The path of the dataset to load
    dataset_category : str
        The category of the dataset to load
        'raw': raw dataset. Contains all data obtained from disruption_py library. All shots, all features, all timeslices
        'train', 'test', 'val': training datasets. Created by running make_training_datasets() on the raw dataset. Filtered to remove bad shots (too short) and bad timeslices (null values, etc.)
    
    Returns
    -------
    data : pandas.DataFrame
        A dataframe containing the data in the dataset
        In the form of data sorted first by shot then by time
    """

    try:
        data = pkgutil.get_data(__name__, f'../data/{device}/{dataset_path}/{dataset_category}.csv')
    except FileNotFoundError:
        data = pkgutil.get_data(__name__, f'data/{device}/{dataset_path}/{dataset_category}.csv')
    data = pd.read_csv(io.BytesIO(data)) # type: ignore

    # Sort by shot number and time
    data = data.sort_values(['shot','time'])
    return data

# def load_dataset_grouped(device, dataset_path, dataset_category):
#     """
#     Load dataset grouped by shot number

#     Parameters
#     ----------
#     device : str
#         The device for which to load the data
#     dataset_path : str
#         The path of the dataset to load
#     dataset_category : str
#         The category of the dataset to load

#     Returns
#     -------
#     group_data : pandas.DataFrameGroupBy
#         A dataframe containing the data in the dataset, grouped by shot number
#     """

#     # Load the raw dataset
#     data = load_dataset(device, dataset_path, dataset_category)

#     # Group the data by shot number
#     group_data = data.groupby('shot')

#     # Return the grouped dataset
#     return group_data

def load_features_outcomes(device, dataset_path, dataset_category, epsilon=1e-4):
    """ Load the specified dataset from a device, and return the features and outcomes
    For usage in SurvivalModel

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
    data : pandas.DataFrame
        A dataframe containing the features for each shot
    """
    
    data = load_dataset(device, dataset_path, dataset_category)

    # Create a parallel outcomes dataframe
    outcomes = data.copy()

    # Create a 'time to last' column for each shot
    outcomes['time_to_last'] = outcomes.groupby('shot')['time'].transform(max) - data['time']
    # Replace zeros in times with low-value epsilon
    outcomes['time_to_last'] = outcomes['time_to_last'].replace(0, epsilon)
    outcomes['time_until_disrupt'] = outcomes['time_until_disrupt'].replace(0, epsilon)
    
    # Outcomes 'event' is 1 if time_until_disrupt is not null (disruptive shot), and 0 otherwise (non-disruptive)
    outcomes['event'] = outcomes['time_until_disrupt'].notnull().astype(int)
    # Outcomes 'time' is time_until_disrupt if it is not null (disruptive shot), and time_to_last otherwise (non-disruptive)
    outcomes['time'] = outcomes['time_until_disrupt'].fillna(outcomes['time_to_last'])

    # make outcomes the correct type for models
    outcomes = outcomes.astype({'event': 'int64', 'time': 'float64'})

    # remove NOT_FEATURES from data
    data = data.drop(NOT_FEATURES, axis=1)
    # trim outcomes to only include event and time columns
    return data, outcomes[['event', 'time']]

# def load_features_events_indicators(device, dataset_path, dataset_category, features):
#     """ Load the specified dataset from a device,
#     and return the features and outcomes and event indicators,
#     grouped by shot number
#     For usage in Recurrent DSM and Recurrent CPH

#     Parameters
#     ----------
#     device : str
#         The device for which to load the data
#     dataset_path : str
#         Path to the dataset to load
#     dataset_group : str
#         The group of dataset to load, either 'raw', 'train', 'test', or 'val'
    
#     Returns
#     -------

#     """

#     data = load_dataset(device, dataset_path, dataset_category)

#     # TODO fill this out after get DPRF working
#     return None

def load_features_labels(device, dataset_path, dataset_category, disruptive_window):
    """Load the features from a dataset and label each timeslice based on some disruptive window.
    In disruptive shots, the timeslice is labeled '1' if it is within the disruptive window, and '0' otherwise.
    In non-disruptive shots, the timeslice is labeled '0' always.
    For usage in binary classifiers

    Parameters
    ----------
    device : str
        The device for which to load the data
    dataset_path : str
        Path to the dataset to load
    dataset_category : str
        The category of the dataset to load
    disruptive_window : float
        The time window to use for labeling disruptive timeslices

    Returns
    -------
    data : pandas.DataFrame
        A dataframe containing the features for each timeslice
    labels : list of int
        A list of labels for each timeslice
        
    """

    data = load_dataset(device, dataset_path, dataset_category)

    # label is '1' if time_until_disrupt is less than disruptive_window, '0' otherwise
    labels = (data['time_until_disrupt'] < disruptive_window).astype(int)

    # remove NOT_FEATURES from data
    data = data.drop(NOT_FEATURES, axis=1)

    return data, labels

def load_feature_list(device, dataset):
    """
    Get all feature names from the dataset.
    Assumes that there are the same features in all created training sets.

    Parameters
    ----------
    device : str
        The device for which to load the data
    dataset : str
        The dataset to load

    Returns
    -------
    features : list of str
        A list of all feature names in the dataset
    """

    # Load the training dataset
    data = load_dataset(device, dataset, 'train')

    # Get the features (ignoring non-feature columns)
    feature_data = data.drop(columns=NOT_FEATURES).columns.values

    # Convert the features to a list
    features = feature_data.tolist()

    # Return the list of features
    return features

def load_shot_list(device, dataset_path, dataset_category):
    """
    Get the list of shots from the dataset
    """

    # Load the raw dataset
    data = load_dataset(device, dataset_path, dataset_category)

    # Find the shots in the dataset
    shots = data['shot'].unique()

    # Return the list of shots
    return shots

def load_disruptive_shot_list(device, dataset_path, dataset_category):
    """
    Get the list of disruptive shots from the dataset
    """

    # Load the raw dataset
    data = load_dataset(device, dataset_path, dataset_category)

    # Find the disruptive shots in the dataset by checking where time_until_disrupt is not null
    disruptive_shots = data[data['time_until_disrupt'].notnull()]['shot'].unique()

    # Return the list of disruptive shots
    return disruptive_shots

def load_non_disruptive_shot_list(device, dataset_path, dataset_category):
    """
    Get the list of non-disruptive shots from the dataset
    """

    # Load the dataset
    data = load_dataset(device, dataset_path, dataset_category)

    # Find the non-disruptive shots in the dataset
    non_disruptive_shots = data[data['time_until_disrupt'].isnull()]['shot'].unique()

    # Return the list of non-disruptive shots
    return non_disruptive_shots
