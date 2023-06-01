import io
import pkgutil

import pandas as pd


def load_dataset(device='cmod', dataset='small_set'):
    """ Load the dataset from the given device and dataset
    """
    
    data = pkgutil.get_data(__name__, 'data/{}/{}.csv'.format(device, dataset))
    data = pd.read_csv(io.BytesIO(data))

    # Add time of last measurement for each shot
    data['last_time'] = data.groupby('shot')['time'].transform('max')

    # Remove the last measurement for each shot
    # TODO: this is a hack to stop negative times from showing up, make better
    data = data[data['time'] != data['last_time']]

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

    # Also return a dataset with features for plotting

    return outcomes, data[train_feats], data[['shot', 'time'] + train_feats]