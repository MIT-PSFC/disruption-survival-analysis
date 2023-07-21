# Class that holds onto data shared between multiple experiments

import numpy as np


from sklearn.metrics import roc_auc_score
from manage_datasets import load_dataset
from experiment_utils import label_shot_data, calculate_alarm_times, clump_many_to_one_statistics

from DisruptionPredictors import DisruptionPredictor

class Experiment:
    """ Class that holds onto data shared between multiple experiments """

    def __init__(self, device, dataset_path, predictor:DisruptionPredictor, name=None, thresholds=np.logspace(-3, 0, 500)):

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

        # Data shared between experiments

        # Dictionaries for true alarms, false alarms, and alarm times for various horizons
        # Key is the horizon in seconds
        # Value is an array of values of shape (num_shots, num_thresholds)
        # All values in these three arrays line up to correspond to the same shots and thresholds
        self.true_alarms = {} 
        self.false_alarms = {}
        self.alarm_times = {}

        # 2D Dictionary of pandas arrays containing risks at each time for each shot
        # First Key is the horizon in seconds
        # Second Key is the shot number
        self.risk_at_times = {}

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
    
    def get_non_disruptive_shot_list(self):
        """ Returns a list of all shots that did not disrupt in the dataset """
        # Get list of shots where 'time_until_disrupt' column has all null values
        return self.all_data[self.all_data['time_until_disrupt'].isnull()]['shot'].unique().astype(int)

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
        risk_at_time = self.get_risk_at_time(shot, horizon)
        return risk_at_time['risk'].values

    def get_risk_at_time(self, shot, horizon):
        """Returns a pandas dataframe containing the risk at each time for a given shot and horizon"""
        try:
            return self.risk_at_times[horizon][shot]
        except KeyError:
            if horizon not in self.risk_at_times:
                self.risk_at_times[horizon] = {}   
            self.risk_at_times[horizon][shot] = self.calc_risk_at_time(shot, horizon)
            return self.risk_at_times[horizon][shot]

    def calc_risk_at_time(self, shot, horizon):
        """ Calculates the risk score for a shot at a given horizon"""
        shot_data = self.all_data[self.all_data['shot'] == shot]
        return self.predictor.calculate_risk_at_time(shot_data, horizon)
    
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
    
    # True Alarms, False Alarms methods

    def get_alarms_times(self, horizon):
        """Attempt to get the true alarms, false alarms, and warning times arrays for a given horizon.
        If they have already been calculated, return them. Otherwise, calculate them and return them.
        """
        try:
            return self.true_alarms[horizon], self.false_alarms[horizon], self.alarm_times[horizon]
        except KeyError:
            self.true_alarms[horizon], self.false_alarms[horizon], self.alarm_times[horizon] = self.calc_alarms_times(horizon)
            return self.true_alarms[horizon], self.false_alarms[horizon], self.alarm_times[horizon]

    def calc_alarms_times(self, horizon):
        """Calculate the true alarms, false alarms, and warning times arrays for a given horizon.
        Where the arrays are of shape (num_shots, num_thresholds)"""

        # Get list of all shots
        shot_list = self.get_shot_list()
        # Get list of disruptive shots
        disruptive_shots = self.get_disruptive_shot_list()
        
        # Create arrays to store the results
        # Array is of shape (num_shots, num_thresholds)
        true_alarms = np.zeros((len(shot_list), len(self.thresholds)))
        false_alarms = np.zeros((len(shot_list), len(self.thresholds)))
        alarm_times = np.zeros((len(shot_list), len(self.thresholds)))

        # Iterate through shots
        for i, shot in enumerate(shot_list):
            disrupt = shot in disruptive_shots

            # Get the alarm times given by the model
            risk_at_time = self.get_risk_at_time(shot, horizon)
            alarm_times_calced = calculate_alarm_times(risk_at_time, self.thresholds)

            # Fill in true and false alarms
            true_alarms[i] = np.array([disrupt and (alarm_time is not None) for alarm_time in alarm_times_calced])
            false_alarms[i] = np.array([(not disrupt) and (alarm_time is not None) for alarm_time in alarm_times_calced])

            # Save the alarm times
            alarm_times[i,:] = alarm_times_calced

        return true_alarms, false_alarms, alarm_times
    
    def true_alarm_rate_vs_threshold(self, horizon):
        """ Get statistics on true alarm rate vs threshold for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one alarm at a given threshold
        """

        true_alarms, _, _ = self.get_alarms_times(horizon)

        true_alarm_rates = np.sum(true_alarms, axis=0) / self.get_num_disruptive_shots()

        return self.thresholds, true_alarm_rates
    
    def false_alarm_rate_vs_threshold(self, horizon):
        """ Get statistics on false alarm rate vs threshold for a given horizon
            This is inherently a macro statistic, since a single shot can have only one alarm at a given threshold
        """

        _, false_alarms, _ = self.get_alarms_times(horizon)

        false_alarm_rates = np.sum(false_alarms, axis=0) / (self.get_num_shots() - self.get_num_disruptive_shots())

        return self.thresholds, false_alarm_rates
     
    # Warning Times Methods

    def get_warning_times_list(self, horizon):
        """Get a list of warning times for each disruptive shot at a given horizon.
        This is a list of arrays of variable length,
        but the arrays will line up such that each index corresponds to the same threshold.
        The list is ordered to be consistent with the list of disruptive shots.
        """
        
        # Get list of all shots
        shot_list = self.get_shot_list()
        # Get list of disruptive shots
        disruptive_shots = self.get_disruptive_shot_list()

        # Get array of alarm times for this horizon
        _, _, alarm_times = self.get_alarms_times(horizon)

        # Create list to store warning times
        warning_times_list = []
        for disruptive_shot in disruptive_shots:
            # Get the time of disruption for this shot
            shot_data = self.all_data[self.all_data['shot'] == disruptive_shot]
            disrupt_time = shot_data['time'].iloc[-1]
            # Find the index of the shot in the shot list
            shot_index = np.where(shot_list == disruptive_shot)[0][0]
            # Get the alarm times for this shot
            shot_alarm_times = alarm_times[shot_index]

            # Calculate the warning times for this shot
            warning_times = np.array([disrupt_time - alarm_time for alarm_time in shot_alarm_times])

            # The only way a NaN will show up in the warning times is if the alarm time was None or Nan
            # If alarm time was None, that means no alarm was raised for this disruptive shot
            # Replace with zero to indicate that there was no warning of disruption
            warning_times = np.array([0 if (warning_time is None) or (np.isnan(warning_time)) else warning_time for warning_time in warning_times])

            warning_times_list.append(warning_times)

        return warning_times_list

    def warning_time_vs_threshold(self, horizon):
        """ Get statistics on warning time vs threshold for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one warning time
        """

        warning_times_list = self.get_warning_times_list(horizon)

        unique_thresholds, mean_warning_times, std_warning_times = clump_many_to_one_statistics(warning_times_list, self.thresholds)

        return unique_thresholds, mean_warning_times, std_warning_times

    def warning_time_vs_true_alarm_rate(self, horizon):
        """ Get statistics on warning time vs true alarm rate for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one warning time
        """

        warning_times_list = self.get_warning_times_list(horizon)
        _, true_alarm_rates = self.true_alarm_rate_vs_threshold(horizon)

        unique_true_alarm_rates, mean_warning_times, std_warning_times = clump_many_to_one_statistics(warning_times_list, true_alarm_rates)

        return unique_true_alarm_rates, mean_warning_times, std_warning_times

    def warning_time_vs_false_alarm_rate(self, horizon):
        """ Get statistics on warning time vs false alarm rate for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one warning time
        """

        warning_times_list = self.get_warning_times_list(horizon)
        _, false_alarm_rates = self.false_alarm_rate_vs_threshold(horizon)

        unique_false_alarm_rates, mean_warning_times, std_warning_times = clump_many_to_one_statistics(warning_times_list, false_alarm_rates)

        # TODO: Ignore zero false positve rate results???
        return unique_false_alarm_rates, mean_warning_times, std_warning_times
    
    def warning_vs_precision(self, horizon):
        """ Get statistics on warning times vs precision for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one warning time
        """

        # TODO need to set this up as a dictionary, so that we can have multiple horizons
        true_alarms, false_alarms, _ = self.get_alarms_times(horizon)

        warning_times_list = self.get_warning_times_list(horizon)

        # Calculate the precision at each threshold
        threshold_precisions = np.sum(true_alarms, axis=0) / (np.sum(true_alarms, axis=0) + np.sum(false_alarms, axis=0))

        unique_precision, mean_warning_times, std_warning_times = clump_many_to_one_statistics(warning_times_list, threshold_precisions)

        # TODO: ignore zero precision results?
        return unique_precision[:], mean_warning_times[:], std_warning_times[:]
