import io
import os
import pkgutil

import numpy as np
import pandas as pd

NOT_FEATURES = ['time', 'shot', 'time_until_disrupt'] # Columns that are not features
DROPPED_FEATURES = ['v_surf'] # Additional features to drop from the raw dataset (should not wind up in the training datasets)

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

    data = pkgutil.get_data(__name__, 'data/{}/{}/{}.csv'.format(device, dataset_path, dataset_category))
    data = pd.read_csv(io.BytesIO(data)) # type: ignore
    # Sort by shot number and time
    data = data.sort_values(['shot','time'])
    return data

def load_dataset_grouped(device, dataset_path, dataset_category):
    """
    Load dataset grouped by shot number

    Parameters
    ----------
    device : str
        The device for which to load the data
    dataset_path : str
        The path of the dataset to load
    dataset_category : str
        The category of the dataset to load

    Returns
    -------
    group_data : pandas.DataFrameGroupBy
        A dataframe containing the data in the dataset, grouped by shot number
    """

    # Load the raw dataset
    data = load_dataset(device, dataset_path, dataset_category)

    # Group the data by shot number
    group_data = data.groupby('shot')

    # Return the grouped dataset
    return group_data

def load_features_outcomes(device, dataset_path, dataset_category, features, epsilon=1e-4):
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

    # trim data to only include features used for training
    # trim outcomes to only include event and time columns
    return data[features], outcomes[['event', 'time']]

def load_features_events_indicators(device, dataset_path, dataset_category, features):
    """ Load the specified dataset from a device,
    and return the features and outcomes and event indicators,
    grouped by shot number
    For usage in Recurrent DSM and Recurrent CPH

    Parameters
    ----------
    device : str
        The device for which to load the data
    dataset_path : str
        Path to the dataset to load
    dataset_group : str
        The group of dataset to load, either 'raw', 'train', 'test', or 'val'
    
    Returns
    -------

    """

    data = load_dataset(device, dataset_path, dataset_category)

    # TODO fill this out after get DPRF working
    return None

def load_features_labels(device, dataset_path, dataset_category, disruptive_window, features):
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
    features : list of str
        The features to load from the dataset

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

    # trim data to only include features used for training
    return data[features], labels

def load_feature_list(device, dataset):
    """
    Get all feature names from the dataset
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

    # Find the disruptive shots in the dataset
    disruptive_shots = data[data['time_until_disrupt'] >= 0]['shot'].unique()

    # Return the list of disruptive shots
    return disruptive_shots

def load_non_disruptive_shot_list(device, dataset_path, dataset_category):
    """
    Get the list of non-disruptive shots from the dataset
    """

    # Load the raw dataset
    data = load_dataset(device, dataset_path, dataset_category)

    # Find the non-disruptive shots in the dataset
    non_disruptive_shots = data[data['time_until_disrupt'].isnull()]['shot'].unique()

    # Return the list of non-disruptive shots
    return non_disruptive_shots


# Functions for making new datasets


def make_training_sets(device, dataset_path, random_seed=0, debug=False):
    """
    Split the raw data into training, test, and validation sets
    """

    # Load the raw dataset
    data = load_dataset(device, dataset_path, 'raw')

    # Eliminate dropped features columns
    data = data.drop(columns=DROPPED_FEATURES)
    # Eliminate timeslices with null values in any feature except time_until_disrupt
    data = data.dropna(subset=[col for col in data.columns if col not in ['time_until_disrupt']])
    # Eliminate timeslices with negative values in time
    data = data[data['time'] >= 0]
    # Remove where time_until_disrupt is negative, keeping where time_until_disrupt is null
    data = data[(data['time_until_disrupt'] >= 0) | (data['time_until_disrupt'].isnull())]
    # Remove shots shorter than 0.5 seconds
    data = data.groupby('shot').filter(lambda x: x['time'].max() - x['time'].min() > 0.5)

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
    train_data.to_csv('data/{}/{}/train.csv'.format(device, dataset_path), index=False)
    test_data.to_csv('data/{}/{}/test.csv'.format(device, dataset_path), index=False)
    val_data.to_csv('data/{}/{}/val.csv'.format(device, dataset_path), index=False)

    # Print the number of shots in each set
    print('Training shots: {}'.format(len(train_shots)))
    print('Test shots: {}'.format(len(test_shots)))
    print('Validation shots: {}'.format(len(val_shots)))

def make_stacked_sets(device, dataset_path, dataset_category, stack_size):
    """Given some dataset, stack the features so each timeslice includes itself and the stack_size previous timeslices
    NOTE: Only use this if the sampling rate of the dataset is constant
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



"""
def load_benchmark_data(predictor:DisruptionPredictor, device, dataset):
    # TODO: this is such a garbo function, completely get rid of it


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
"""