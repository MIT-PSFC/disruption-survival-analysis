# Class that holds onto data shared between multiple experiments

import numpy as np


from sklearn.metrics import roc_auc_score
from preprocess_datasets import load_dataset
from evaluate_performance import label_shot_data

from DisruptionPredictors import DisruptionPredictor

class Experiment:

    def __init__(self, device, dataset, predictor:DisruptionPredictor, name=None):

        # feature_data: only what is fed to predictor
        # all_data: all data, including shot, time, time_until_disrupt, and features fed to predictor

        self.device = device
        self.dataset = dataset
        self.predictor = predictor

        # Load data
        self.all_data = load_dataset(device, dataset)
        
        # Transform required features using the predictor's transformer, discard the rest
        self.feature_data = predictor.transformer.transform(self.all_data[predictor.features])

        # Remove the features that were not used, keep other important columns
        self.all_data = self.all_data[['shot', 'time', 'time_until_disrupt']]

        # Replace the old data with the transformed data
        self.all_data[predictor.features] = self.feature_data

        # Set the name of the experiment
        if name is None:
            self.name = device + '_' + dataset

    def get_shot_list(self):
        """ Returns a list of all shots in the dataset """
        return self.all_data['shot'].unique()
    
    def get_disruptive_shot_list(self):
        """ Returns a list of all shots that disrupted in the dataset """
        return self.all_data[self.all_data['time_until_disrupt'] >= 0]['shot'].unique()
    

    # ROC AUC methods

    def roc_auc_single(self, horizons, shot):
        """ Returns the ROC AUC on a shot basis over many horizons 
        ONLY defined for disruptive shots
        """

        # Determine if shot was disruptive
        disruptive = shot in self.get_disruptive_shot_list()
        if not disruptive:
            raise ValueError('Shot was not disruptive, cannot calculate ROC AUC because only one class present in y_true.')

        # Get the features for the shot
        shot_data = self.all_data[self.all_data['shot'] == shot]
        
        roc_auc_list = []   # List of ROC AUCs corresponding to each horizon
        for horizon in horizons:
            # Set up true labels and predicted risk scores
            y_true = label_shot_data(shot_data, disruptive, horizon)
            y_pred = self.predictor.calculate_risk(shot_data, horizon).values

            roc_auc_list.append(roc_auc_score(y_true, y_pred))

        return roc_auc_list
    
    def roc_auc_macro(self, horizons):
        """ Returns the ROC AUC for the dataset averaged over all disruptive shots """
    
        # Get a list of all disruptive shots in the dataset
        shot_list = self.get_disruptive_shot_list()

        # Iterate through all shots and calculate the ROC AUC for each
        roc_auc_array = np.zeros((len(shot_list), len(horizons)))
        for i, shot in enumerate(shot_list):
            roc_auc_array[i,:] = self.roc_auc_single(horizons, shot)

        # Average the ROC AUCs over all shots
        return np.mean(roc_auc_array, axis=0), np.std(roc_auc_array, axis=0)
    

    def roc_auc_micro_all(self, horizons, disrupt_only=False):
        """ Returns the ROC AUC on a timeslice basis over many horizons"""

        # Get list of shots to use
        if disrupt_only:
            shot_list = self.get_disruptive_shot_list()
        else:
            shot_list = self.get_shot_list()

        roc_auc_list = []   # List of ROC AUCs corresponding to each horizon
        for horizon in horizons:
            y_true = []
            y_pred = []

            for shot in shot_list:
                # Get the features for the shot
                shot_data = self.all_data[self.all_data['shot'] == shot]
                # Determine if shot was disruptive
                disruptive = shot in self.get_disruptive_shot_list()
                # Set up true labels and predicted risk scores
                y_true.extend(label_shot_data(shot_data, disruptive, horizon))
                y_pred.extend(self.predictor.calculate_risk(shot_data, horizon).values)

            roc_auc_list.append(roc_auc_score(y_true, y_pred))

        return roc_auc_list