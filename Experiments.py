# Class that holds onto data shared between multiple experiments

import numpy as np


from sklearn.metrics import roc_auc_score
from manage_datasets import load_dataset
from experiment_utils import label_shot_data, calculate_disruption_time

from DisruptionPredictors import DisruptionPredictor

class Experiment:
    """ Class that holds onto data shared between multiple experiments """

    # Data shared between experiments

    # Dictionaries for true alarms, false alarms, and warning times for various horizons
    # Key is the horizon in seconds
    # Value is an array of values of shape (num_shots, num_thresholds)
    # All values in these three arrays line up to correspond to the same shots and thresholds
    true_alarms = {} 
    false_alarms = {}
    warning_times = {}

    # 2D Dictionary of pandas arrays containing risks at each time for each shot
    # First Key is the horizon in seconds
    # Second Key is the shot number
    risk_at_times = {}

    def __init__(self, device, dataset_path, predictor:DisruptionPredictor, name=None, thresholds=np.logspace(-3, 0, 1000)):

        # all_data: all data, including shot, time, time_until_disrupt, and features fed to predictor

        self.device = device
        self.dataset_path = dataset_path
        self.predictor = predictor

        # Load test data
        self.all_data = load_dataset(device, dataset_path, 'test')
        
        # Transform required features using the predictor's transformer, discard the rest
        feature_data = predictor.transformer.transform(self.all_data[predictor.features])

        # Remove the features that were not used, keep other important columns
        self.all_data = self.all_data[['shot', 'time', 'time_until_disrupt']]

        # Replace the old data with the transformed data
        self.all_data[predictor.features] = feature_data

        # Set the name of the experiment
        if name is None:
            self.name = device + ' ' + dataset_path
        else:
            self.name = name

        # Set the thresholds for usage in tpr/fpr calculations
        self.thresholds = thresholds

    def get_shot_list(self):
        """ Returns a list of all shots in the dataset """
        # If shot list has already been calculated, return it
        try:
            return self.shot_list
        except:
            self.shot_list = self.all_data['shot'].unique().astype(int)
            return self.shot_list
    
    def get_disruptive_shot_list(self):
        """ Returns a list of all shots that disrupted in the dataset """
        return self.all_data[self.all_data['time_until_disrupt'] >= 0]['shot'].unique().astype(int)
    
    def get_num_shots(self):
        """ Returns the number of shots in the dataset """
        return len(self.get_shot_list())
    
    def get_num_disruptive_shots(self):
        """ Returns the number of disruptive shots in the dataset """
        return len(self.get_disruptive_shot_list())
    
    def get_time(self, shot):
        """ Returns the times for a given shot """
        shot_data = self.all_data[self.all_data['shot'] == shot]
        return shot_data['time'].values

    def get_risk(self, shot, horizon):
        """ Returns the risk score for a shot at a given horizon """
        try:
            return self.risk_at_times[horizon][shot]
        except KeyError:
            if horizon not in self.risk_at_times:
                self.risk_at_times[horizon] = {}   
            self.risk_at_times[horizon][shot] = self.calc_risk(shot, horizon)
            return self.risk_at_times[horizon][shot]
    
    def calc_risk(self, shot, horizon):
        """ Calculates the risk score for a shot at a given horizon"""
        shot_data = self.all_data[self.all_data['shot'] == shot]
        return self.predictor.calculate_risk_at_time(shot_data, horizon)['risk'].values
    
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
            y_pred = self.predictor.calculate_risk_at_time(shot_data, horizon)['risk'].values

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
                y_pred.extend(self.predictor.calculate_risk_at_time(shot_data, horizon)['risk'].values)

            roc_auc_list.append(roc_auc_score(y_true, y_pred))

        return roc_auc_list
    
    # TPR, FPR, Warning Time methods

    def get_alarms_times(self, horizon):
        """Attempt to get the true alarms, false alarms, and warning times arrays for a given horizon.
        If they have already been calculated, return them. Otherwise, calculate them and return them.
        """
        try:
            return self.true_alarms[horizon], self.false_alarms[horizon], self.warning_times[horizon]
        except KeyError:
            self.true_alarms[horizon], self.false_alarms[horizon], self.warning_times[horizon] = self.calc_alarms_times(horizon)
            return self.true_alarms[horizon], self.false_alarms[horizon], self.warning_times[horizon]

    def calc_alarms_times(self, horizon):
        """Calculate the true alarms, false alarms, and warning times arrays for a given horizon.
        Where the arrays are of shape (num_shots, num_thresholds)"""

        # Get list of all shots
        shot_list = self.get_shot_list()
        # Get list of disruptive shots
        disruptive_shots = self.get_disruptive_shot_list()
        
        # Create arrays to store the results
        # Array is of shape (num_shots, num_thresholds)
        true_positives = np.zeros((len(shot_list), len(self.thresholds)))
        false_positives = np.zeros((len(shot_list), len(self.thresholds)))

        # Create list to store warning times
        # This is a list of arrays of variable length,
        # but the arrays will line up such that each index corresponds to the same threshold
        warning_times = []

        # Iterate through shots
        for i, shot in enumerate(shot_list):
            disrupt = shot in disruptive_shots
            shot_data = self.all_data[self.all_data['shot'] == shot]

            # Get the disruption time predicted by the model
            risk_at_time = self.get_risk(shot, horizon)
            predicted_times = calculate_disruption_time(risk_at_time, self.thresholds)

            # Fill in true and false positives
            true_positives[i] = np.array([disrupt and (predicted_time is not None) for predicted_time in predicted_times])
            false_positives[i] = np.array([(not disrupt) and (predicted_time is not None) for predicted_time in predicted_times])

            # If shot is disruptive, can fill in Time to First True Detection
            if disrupt:
                # Find actual disruption time by looking at last time in shot
                true_time = shot_data['time'].iloc[-1]

                warning_times.append(np.array([true_time - predicted_time for predicted_time in predicted_times if predicted_time is not None]))

        return true_positives, false_positives, warning_times
    
    def tpr_vs_threshold(self, horizon):
        """ Get statistics on true positive rate vs threshold for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one true positive rate
        """

        true_positives, _, _ = self.get_alarms_times(horizon)

        true_positive_rates = np.sum(true_positives, axis=0) / self.get_num_disruptive_shots()

        return self.thresholds, true_positive_rates
    
    def fpr_vs_threshold(self, horizon):
        """ Get statistics on false positive rate vs threshold for a given horizon
            This is inherently a macro statistic, since a single shot can have only one false positive rate
        """

        _, false_positives, _ = self.get_alarms_times(horizon)

        false_positive_rates = np.sum(false_positives, axis=0) / (self.get_num_shots() - self.get_num_disruptive_shots())

        return self.thresholds, false_positive_rates
    
    def warning_vs_threshold(self, horizon):
        """ Get statistics on warning times vs threshold for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one warning time
        """

        _, _, warning_times = self.get_alarms_times(horizon)

        mean_warning_times = []
        std_warning_times = []

        # Calculate the average warning time for each threshold
        for i in range(len(self.thresholds)):

            clump_warning_times = []

            for warning_time in warning_times:
                try:
                    clump_warning_times.append(warning_time[i])
                except IndexError:
                    # This is a disruptive shot that didn't have a detection at this threshold
                    # Warning time is 0
                    clump_warning_times.append(0)

            mean_warning_times.append(np.mean(clump_warning_times))
            std_warning_times.append(np.std(clump_warning_times))

        return self.thresholds, mean_warning_times, std_warning_times

    def warning_vs_fpr(self, horizon):
        """ Get statistics on warning times vs FPR for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one warning time
        """

        # TODO need to set this up as a dictionary, so that we can have multiple horizons
        _, false_positives, warning_times = self.get_alarms_times(horizon)

        mean_warning_times = []
        std_warning_times = []

        # Calculate the false positive rate at each threshold
        false_positive_rates = np.sum(false_positives, axis=0) / (self.get_num_shots() - self.get_num_disruptive_shots())

        # TODO: Should really really vectorize this

        # Calculate the average warning time for each false positive rate
        fpr_times = []
        for i in range(len(self.thresholds)):
        
            for warning_time in warning_times:
                try:
                    fpr_times.append(warning_time[i])
                except IndexError:
                    # This is a disruptive shot that didn't have a detection at this threshold
                    # Warning time is 0
                    fpr_times.append(0)
            
            # Clump the detection times that share a false positive rate together
            # Or if we're at the end, we need to add the last one regardless
            if i == len(self.thresholds) - 1 or (false_positive_rates[i] != false_positive_rates[i+1]):
                if len(fpr_times) > 0:
                    mean_warning_times.append(np.mean(fpr_times))
                    std_warning_times.append(np.std(fpr_times))
                    fpr_times = []
                else:
                    # If there are no detection times, that means false positive rate is 0. Detection time is 0.
                    mean_warning_times.append(0)
                    std_warning_times.append(0)
                

        # Eliminate duplicate false positive rates.
        # However, this sorts the false positive rates, so we need to reverse the order afterwards
        unique_false_positive_rates = np.unique(false_positive_rates)
        # Reverse the order so that the false positive rates are increasing (to once again line up with the detection times)
        unique_false_positive_rates = unique_false_positive_rates[::-1]

        # Ignore zero false positve rate results
        return unique_false_positive_rates[:-1], mean_warning_times[:-1], std_warning_times[:-1]
    
    def warning_vs_precision(self, horizon):
        """ Get statistics on warning times vs precision for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one warning time
        """

        # TODO need to set this up as a dictionary, so that we can have multiple horizons
        true_positives, false_positives, warning_times = self.get_alarms_times(horizon)

        mean_warning_times = []
        std_warning_times = []

        # Calculate the precision at each threshold
        threshold_precisions = np.sum(true_positives, axis=0) / (np.sum(true_positives, axis=0) + np.sum(false_positives, axis=0))

        # Calculate the average warning time for each precision
        # Each threshold has its own precision which is not necessarily unique
        # Unlike FPR, precision is not is not monatonically decreasing with threshold
        # Need to keep careful track of the indices while calculating this.

        # Eliminate duplicate precision.
        unique_precision = np.unique(threshold_precisions)

        for precision in unique_precision:
            precision_times = []

            # Find the indices of the thresholds with this precision
            precision_indices = np.where(threshold_precisions == precision)[0]

            for i in precision_indices:
                for warning_time in warning_times:
                    try:
                        precision_times.append(warning_time[i])
                    except IndexError:
                        # This is a disruptive shot that didn't have a detection at this threshold
                        # Warning time is 0
                        precision_times.append(0)

            # Clump the detection times that share a precision together
            # This list should line up with the unique_precision list
            mean_warning_times.append(np.mean(precision_times))
            std_warning_times.append(np.std(precision_times))

        # TODO: ignore zero precision results?
        return unique_precision[:], mean_warning_times[:], std_warning_times[:]
