import io
import pkgutil

import numpy as np
import pandas as pd

from DisruptionPredictors import DisruptionPredictor

# Raw dataset: contains all data obtained from disruption_py library. All shots, all features, all timeslices
# train, test, and validation datasets: created by running make_training_datasets() on the raw dataset
#   Does some filtering to remove bad shots and bad timeslices (null values, shots very short, etc.)

NOT_FEATURES = ['time', 'shot', 'time_until_disrupt'] # Columns that are not features
DROPPED_FEATURES = ['v_surf'] # Additional features to drop from the raw dataset (should not wind up in the training datasets)

def load_dataset(device, dataset_path, dataset_category):
    """ Load a dataset from a specific device, sorted by shot number and time
    Does not do any further processing.

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
    data = pkgutil.get_data(__name__, 'data/{}/{}/{}.csv'.format(device, dataset_path, dataset_category))
    data = pd.read_csv(io.BytesIO(data)) # type: ignore
    # Sort by shot number and time
    data = data.sort_values(['shot','time'])
    return data

def create_outcomes(device, dataset_path, dataset_category, epsilon=1e-4):
    """ Adds a 'time_to_last' column to the dataset
    Replace zeroes in time_until_disruption with epsilon

    Intended to be use
    
    """
    
    data = load_dataset(device, dataset_path, dataset_category)

    # Create a 'time to last measurement' column for each shot
    # This is the time from the present time to the last measurement
    data['time_to_last'] = data.groupby('shot')['time'].transform(max) - data['time']

    # Replace zeros in times with low-value epsilon
    data['time_to_last'] = data['time_to_last'].replace(0, epsilon)
    data['time_until_disrupt'] = data['time_until_disrupt'].replace(0, epsilon)

    return data

def load_features_outcomes(device, dataset_path, dataset_category, features):
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
    """
    
    data = load_dataset(device, dataset_path, dataset_category)

    outcomes = pd.DataFrame()
    
    # Outcomes 'event' is 1 if time_until_disrupt is not null
    outcomes['event'] = data['time_until_disrupt'].notnull().astype(int)

    # Outcomes 'time' is time_until_disrupt if it is not null, and time_to_last otherwise
    outcomes['time'] = data['time_until_disrupt'].fillna(data['time_to_last'])

    # make outcomes is the correct type
    outcomes = outcomes.astype({'event': 'int64', 'time': 'float64'})

    # trim data to only include features used for training

    return data[features], outcomes


def load_features_events_indicators(device, dataset_path, dataset_category, features):
    """ For usage in Recurrent DSM and Recurrent CPH
    Load the specified dataset from a device,
    and return the features and outcomes and event indicators,
    grouped by shot number

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
    outcomes : pandas.DataFrame
        A dataframe containing the outcomes for each shot
        Outcomes are whether or not the event occurred, and the time until the event or last measurement
    features : pandas.DataFrame
        A dataframe containing the features for each shot
    """

    data = load_dataset(device, dataset_path, dataset_category)

    # TODO fill this out after get DPRF working
    return None


def load_features_labels(device, dataset_path, dataset_category, cutoff_threshold, features=None):
    """For usage in Random Forest Classifier
    
    """

    data = load_dataset(device, dataset_path, dataset_category)

    # label is '1' if time_until_disrupt is less than cutoff_threshold, '0' otherwise
    labels = (data['time_until_disrupt'] < cutoff_threshold).astype(int)

    # trim data to only include features used for training
    return data[features], labels

def make_training_sets(device, dataset_path, random_seed=0, window=None):
    """
    Split the raw data into training, test, and validation sets
    """

    # Load the raw dataset
    data = load_dataset(device, dataset_path, 'raw')

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

def get_shot_list(device, dataset_path, dataset_category):
    """
    Get the list of shots from the dataset
    """

    # Load the raw dataset
    data = load_dataset(device, dataset_path, dataset_category)

    # Find the shots in the dataset
    shots = data['shot'].unique()

    # Return the list of shots
    return shots

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

def make_stacked_sets(device, dataset, stack_size):
    """Given some dataset, stack the features so each row includes itself and the k previous rows
    This assumes that the sampling time of the dataset is constant
    """

    data = load_dataset(device, dataset)

    # Get the columns to stack
    features = load_features(device, dataset)

    extended_features = features.copy()
    for feature in features:
        for i in range(stack_size):
            extended_features.append(f'{feature}_{i}')

    stacked_features = pd.DataFrame(columns=extended_features)

    # Get the shot numbers
    shot_numbers = data['shot'].unique()

    # For each shot, stack the features

    for shot in shot_numbers:

        shot_stack = data[data['shot'] == shot].copy()

        for feature in features:
            for i in range(stack_size):
                shot_stack[f'{feature}_{i}'] = shot_stack[feature].shift(stack_size)

        # TODO: replace new NaN's with -1's and let the model learn how to deal with it
        #shot_stack = shot_stack.dropna(subset=features)

        stacked_features = pd.concat([stacked_features, shot_stack], ignore_index=True)
    
    # Remove '_train', '_test', or '_val' from the dataset name
    # But keep track of whether it was a train, test, or val dataset
    dataset_folder = dataset.split('/')[0]
    dataset_type = dataset.split('/')[1]

    # Save stacked features under new name
    stacked_features.to_csv(f'data/{device}/{dataset_folder}/{stack_size}_stack/{dataset_type}.csv', index=False)

def load_features(device, dataset):
    """
    Get all feature names from the dataset
    """

    # Load the raw dataset
    data = load_dataset(device, dataset)

    # Get the features
    feature_data = data.drop(columns=NOT_FEATURES).columns.values

    # Convert the features to a list
    features = feature_data.tolist()

    # Return the list of features
    return features
        