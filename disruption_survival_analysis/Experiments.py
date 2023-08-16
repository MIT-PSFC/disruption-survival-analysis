# Class that holds onto data shared between multiple experiments

import numpy as np

from sklearn.metrics import roc_auc_score
from disruption_survival_analysis.manage_datasets import load_dataset
from disruption_survival_analysis.experiment_utils import label_shot_data, calculate_alarm_times, calculate_alarm_times_hysteresis, calculate_alarm_times_ettd, timeslice_micro_average, area_under_curve, calculate_f1_scores, expected_time_to_disruption_integral, clump_many_to_one_statistics
from disruption_survival_analysis.model_utils import get_model_for_experiment, name_model

from auton_survival.estimators import SurvivalModel # CPH, DCPH, DSM, DCM, RSF
from sklearn.ensemble import RandomForestClassifier

from disruption_survival_analysis.DisruptionPredictors import DisruptionPredictorSM, DisruptionPredictorRF

class Experiment:
    """ Class that holds onto data shared between multiple experiments """

    def __init__(self, config, experiment_type):
        """
        Make an experiment from a config dictionary. 
        If the experiment type is 'test', then the experiment will be a test experiment.
        If the experiment type is 'val', then the experiment will be a validation experiment.

        Parameters
        ----------
        config : dict
            Dictionary of everything unique to this experiment.
            Should contain the model type, the metric to be evaluated, which dataset to use, and some model-specific hyperparameters
        experiment_type : str
            The type of experiment to make. Either 'test' or 'val'
            
        Returns
        -------
        experiment : Experiment
            The experiment to be run
        """

        # Get some info from the config
        self.device = config['aa-device']
        self.dataset_path = config['aa-dataset-path']

        # Set the type of experiment
        self.experiment_type = experiment_type
        # Load data for experiment: all data, including shot, time, time_until_disrupt, and features fed to predictor
        self.all_data = load_dataset(self.device, self.dataset_path, experiment_type)

        # Create the model and predictor for the experiment
        model = get_model_for_experiment(config, experiment_type)
        required_warning_time = config['ab-required-warning-time']
        self.name = name_model(config)

        if isinstance(model, SurvivalModel):
            self.predictor = DisruptionPredictorSM(self.name, model, required_warning_time, config['horizon'])
        elif isinstance(model, RandomForestClassifier):
            self.predictor = DisruptionPredictorRF(self.name, model, required_warning_time, config['class_time'])
        else:
            raise ValueError('Model type not recognized')
        
        # Get the alarm type from the config
        self.alarm_type = config['ab-alarm-type']

        # Set the thresholds for usage in tpr/fpr calculations
        if self.alarm_type == 'sthr':
            # Simple Threshold
            self.thresholds = np.linspace(0, 1, 100)
        elif self.alarm_type == 'hyst':
            # Hysteresis
            # Make list of tuples of (min, max, time) for hysteresis
            # where max goes from 0.1 to 1 and min goes from 0 to max
            # and trigger time goes from 0 to 80 ms
            self.thresholds = []
            for max in np.linspace(0.1, 1, 9):
                for min in np.linspace(0, max, 4):
                    for time in np.linspace(0, 0.08, 4):
                        self.thresholds.append((min, max, time))
        elif self.alarm_type == 'ettd':
            # Expected time to disruption
            self.thresholds = [0.1, 0.02] # Expected time to disruption thresholds in seconds
        else:
            raise ValueError('Invalid alarm_type: ' + self.alarm_type)

        # Dictionaries for true alarms, false alarms, and alarm times for various horizons
        # Alarm time key is the horizon in seconds
        # True alarm and False alarm keys are first the horizon in seconds, then the required warning time in seconds
        # Value is an array of values of shape (num_shots, num_thresholds)
        # All values in these three arrays line up to correspond to the same shots and thresholds
        self.alarm_times = {}
        self.true_alarms = {} 
        self.false_alarms = {}

    # Simple helper methods

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
    
    def get_num_non_disruptive_shots(self):
        """ Returns the number of non-disruptive shots in the dataset """
        return len(self.get_non_disruptive_shot_list())
    
    def get_time(self, shot):
        """ Returns the times for a given shot """
        shot_data = self.all_data[self.all_data['shot'] == shot]
        return shot_data['time'].values
    
    def get_shot_duration(self, shot):
        """ Returns the duration of a given shot """
        shot_data = self.all_data[self.all_data['shot'] == shot]
        return shot_data['time'].max() - shot_data['time'].min()

    def get_all_shot_durations(self):
        """ Returns the duration of each shot in the dataset """
        shot_durations = []
        for shot in self.get_shot_list():
            shot_durations.append(self.get_shot_duration(shot))
        return np.array(shot_durations)
    
    def get_disruptive_shot_durations(self):
        """ Returns the duration of each disruptive shot in the dataset """
        shot_durations = []
        for shot in self.get_disruptive_shot_list():
            shot_durations.append(self.get_shot_duration(shot))
        return np.array(shot_durations)
    
    def get_non_disruptive_shot_durations(self):
        """ Returns the duration of each non-disruptive shot in the dataset """
        shot_durations = []
        for shot in self.get_non_disruptive_shot_list():
            shot_durations.append(self.get_shot_duration(shot))
        return np.array(shot_durations)

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
            y_pred = self.predictor.get_risk(shot, shot_data, horizon=horizon)

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
                new_labels = label_shot_data(shot_data, disruptive, horizon)
                new_predictions = self.predictor.get_risk(shot, shot_data, horizon=horizon)
                y_true.extend(new_labels)
                y_pred.extend(new_predictions)

            roc_auc_list.append(roc_auc_score(y_true, y_pred))

        return roc_auc_list
    
    # True Alarms, False Alarms methods

    def get_alarm_times(self, horizon=None):
        """Attempt to get the alarm times arrays for a given horizon.
        If they have already been calculated, return them. Otherwise, calculate them and return them.
        """

        if horizon is None:
            horizon = self.predictor.trained_disruptive_window

        if horizon not in self.alarm_times:
            self.alarm_times[horizon] = self._calc_alarm_times(horizon)

        return self.alarm_times[horizon]
        
    def _calc_alarm_times(self, horizon):
        """Calculate the alarm times for a given horizon using the Experiment's alarm type.
        Where the arrays are of shape (num_shots, num_thresholds)"""

        # Get list of all shots
        shot_list = self.get_shot_list()
        
        # Create arrays to store the results
        # Array is of shape (num_shots, num_thresholds)
        alarm_times = np.zeros((len(shot_list), len(self.thresholds)))

        # Determine which function to use 
        if self.alarm_type == 'sthr':
            # Simple Threshold

            # Iterate through shots
            for i, shot in enumerate(shot_list):
                shot_data = self.all_data[self.all_data['shot'] == shot]

                # Get the alarm times given by the model
                risk_at_time = self.predictor.get_risk_at_times(shot, shot_data, horizon)
                alarm_times_calced = calculate_alarm_times(risk_at_time, self.thresholds)

                # Save the alarm times
                alarm_times[i,:] = alarm_times_calced
        
        elif self.alarm_type == 'hyst':
            # Hysteresis

            # Iterate through shots
            for i, shot in enumerate(shot_list):
                shot_data = self.all_data[self.all_data['shot'] == shot]

                # Get the alarm times given by the model
                risk_at_time = self.predictor.get_risk_at_times(shot, shot_data, horizon)
                alarm_times_calced = calculate_alarm_times_hysteresis(risk_at_time, self.thresholds)

                # Save the alarm times
                alarm_times[i,:] = alarm_times_calced
        
        elif self.alarm_type == 'ettd':
            # Expected Time to Disruption

            # Iterate through shots
            for i, shot in enumerate(shot_list):
                shot_data = self.all_data[self.all_data['shot'] == shot]
                
                # Get the expected time to disruption of the shot
                expected_lifetime = self.predictor.get_ettd_at_times(shot, shot_data)
                alarm_times_calced = calculate_alarm_times_ettd(expected_lifetime, self.thresholds)

                # Save the alarm times
                alarm_times[i,:] = alarm_times_calced

        elif self.alarm_type == 'ethy':
            # Expected time to disruption hysteresis

            # TBD if we want to do this (probably yes, and it shouldn't be too difficult)
            pass

        else:
            raise ValueError(f'Unknown alarm type: {self.alarm_type}')

        return alarm_times

    def get_true_false_alarms(self, horizon=None, required_warning_time=None):
        """Attempt to get the true alarms and false alarms arrays for a given horizon and required warning time.
        If they have already been calculated, return them. Otherwise, calculate them and return them."""

        if horizon is None:
            horizon = self.predictor.trained_disruptive_window
        if required_warning_time is None:
            required_warning_time = self.predictor.trained_required_warning_time

        if horizon not in self.true_alarms or horizon not in self.false_alarms:
            self.true_alarms[horizon] = {}
            self.false_alarms[horizon] = {}
        if required_warning_time not in self.true_alarms[horizon]:
            self.true_alarms[horizon][required_warning_time], self.false_alarms[horizon][required_warning_time] = self._calc_true_false_alarms(horizon, required_warning_time)

        return self.true_alarms[horizon][required_warning_time], self.false_alarms[horizon][required_warning_time]

    def _calc_true_false_alarms(self, horizon, required_warning_time):
        """Calculate the true alarms and false alarm arrays for a given horizon and required warning time,
        where warning time is the time before the disruption that the alarm must be raised.
        Where the arrays are of shape (num_shots, num_thresholds)"""

        # Get list of all shots
        shot_list = self.get_shot_list()
        # Get list of disruptive shots
        disruptive_shots = self.get_disruptive_shot_list()
        # Create arrays to store the results
        # Array is of shape (num_shots, num_thresholds)
        true_alarms = np.zeros((len(shot_list), len(self.thresholds)))
        false_alarms = np.zeros((len(shot_list), len(self.thresholds)))

        # Get the alarm times given by the model
        alarm_times = self.get_alarm_times(horizon)

        # Iterate through shots
        for i, shot in enumerate(shot_list):
            disrupt = shot in disruptive_shots

            # If the shot is disruptive, find the time of disruption
            # Even if the shot is non-disruptive, we will determine a warning time for the false alarm calculation
            disruption_time = self.get_shot_duration(shot)
            # Warning time is disruption time minus alarm time
            warning_times = disruption_time - alarm_times[i,:]

            # Fill in true and false alarms
            # True alarm is when shot is disruptive and warning time is greater than required warning time
            # False alarm is when shot is not disruptive but an alarm time is still given
            true_alarms[i,:] = (disrupt & (warning_times >= required_warning_time)).astype(int)
            false_alarms[i,:] = (~disrupt & ~np.isnan(alarm_times[i,:])).astype(int)
            
        return true_alarms, false_alarms
    
    def true_false_alarm_rates(self, horizon=None, required_warning_time=None):
        """ Get statistics on false alarm rate vs threshold for a given horizon and required warning time
            This is inherently a macro statistic, since a single shot can have only one alarm at a given threshold
        """

        true_alarms, false_alarms = self.get_true_false_alarms(horizon, required_warning_time)

        true_alarm_rates = np.sum(true_alarms, axis=0) / self.get_num_disruptive_shots()
        false_alarm_rates = np.sum(false_alarms, axis=0) / (self.get_num_shots() - self.get_num_disruptive_shots())

        return true_alarm_rates, false_alarm_rates

    def true_alarm_rate_vs_false_alarm_rate(self, horizon=None, required_warning_time=None):

        true_alarm_rates, false_alarm_rates = self.true_false_alarm_rates(horizon, required_warning_time)

        unique_false_alarms, avg_true_alarm_rates, std_true_alarm_rates = clump_many_to_one_statistics(false_alarm_rates, true_alarm_rates)

        return unique_false_alarms, avg_true_alarm_rates#, std_true_alarm_rates
     
    def missed_alarm_rate_vs_false_alarm_rate(self, horizon=None, required_warning_time=None):

        true_alarms, false_alarms = self.get_true_false_alarms(horizon, required_warning_time)

        missed_alarm_rates = 1 - np.sum(true_alarms, axis=0) / self.get_num_disruptive_shots()
        false_alarm_rates = np.sum(false_alarms, axis=0) / (self.get_num_shots() - self.get_num_disruptive_shots())

        unique_false_alarms, avg_missed_alarm_rates, std_missed_alarm_rates = clump_many_to_one_statistics(false_alarm_rates, missed_alarm_rates)

        return unique_false_alarms, avg_missed_alarm_rates

    # Warning Times Methods

    def get_warning_times_list(self, horizon=None):
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
        alarm_times = self.get_alarm_times(horizon)

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

    def warning_time_vs_threshold(self, horizon=None):
        """ Get statistics on warning time vs threshold for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one warning time
        """

        warning_times_list = self.get_warning_times_list(horizon)

        unique_thresholds, avg_warning_times, std_warning_times = clump_many_to_one_statistics(self.thresholds, warning_times_list)

        return unique_thresholds, avg_warning_times, std_warning_times

    def warning_time_vs_false_alarm_rate(self, horizon, required_warning_time):
        """ Get statistics on warning time vs false alarm rate for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one warning time
        """

        warning_times_list = self.get_warning_times_list(horizon)
        _, false_alarm_rates = self.true_false_alarm_rates(horizon, required_warning_time)

        unique_false_alarm_rates, avg_warning_times, std_warning_times = clump_many_to_one_statistics(false_alarm_rates, warning_times_list)

        # TODO: Ignore zero false positve rate results???
        return unique_false_alarm_rates, avg_warning_times, std_warning_times
    
    # Metrics methods

    def evaluate_metric(self, metric_type, horizon=None, required_warning_time=None):
        """
        Evaluate a metric on the experiment

        Parameters
        ----------
        metric_type : str
            The type of metric to evaluate
            Options are 'tslic', 'auroc', 'auwtc', 'maxf1', 'ettdi'
        horizon : int, optional
            The horizon to evaluate the metric at
            If None, use the horizon the model was hyperparameter tuned for
        required_warning_time : float, optional
            The required warning time to evaluate the metric at
            If None, use the required warning time the model was trained on

        Returns
        -------
        metric_val : float
            The value of the metric
        """

        if metric_type == 'tslic':
            # Timeslice metric. Micro avgerage over entire dataset
            metric_val = timeslice_micro_average(self.device, self.dataset_path, self.predictor.model, self.experiment_type)
        elif metric_type == 'auroc':
            # Area under ROC curve
            false_alarm_rates, true_alarm_rates = self.true_alarm_rate_vs_false_alarm_rate(horizon, required_warning_time)
            metric_val = area_under_curve(false_alarm_rates, true_alarm_rates)
        elif metric_type == 'auwtc':
            # Area under warning time curve
            false_alarm_rates, warning_times, _ = self.warning_time_vs_false_alarm_rate(horizon, required_warning_time)
            metric_val = area_under_curve(false_alarm_rates, warning_times)
        elif metric_type == 'maxf1':
            # Highest f1 score over all the thresholds
            true_alarms, false_alarms = self.get_true_false_alarms(horizon, required_warning_time)

            true_alarm_count_array = np.sum(true_alarms, axis=0)
            false_alarm_count_array = np.sum(false_alarms, axis=0)

            num_disruptive_shots = self.get_num_disruptive_shots()

            f1_scores = calculate_f1_scores(true_alarm_count_array, false_alarm_count_array, num_disruptive_shots)

            metric_val = np.max(f1_scores)

        elif metric_type == 'ettdi':
            # Expected time to disruption error integral
            metric_val = expected_time_to_disruption_integral()
        else:
            metric_val = None

        return metric_val
    
    def max_f1_info(self, horizon=None, required_warning_time=None):
        # Get related info for the ebst f1 score

        true_alarms, false_alarms = self.get_true_false_alarms(horizon, required_warning_time)

        true_alarm_count_array = np.sum(true_alarms, axis=0)
        false_alarm_count_array = np.sum(false_alarms, axis=0)

        num_disruptive_shots = self.get_num_disruptive_shots()

        f1_scores = calculate_f1_scores(true_alarm_count_array, false_alarm_count_array, num_disruptive_shots)

        # Find the index of the best f1 score
        best_f1_score_index = np.argmax(f1_scores)

        # Get the true alarm rate, false alarm rate, and warning time at the best f1 score
        true_alarm_rate = true_alarm_count_array[best_f1_score_index]/num_disruptive_shots
        false_alarm_rate = false_alarm_count_array[best_f1_score_index]/num_disruptive_shots
        
        # Get threshold at the best f1 score
        best_f1_threshold = self.thresholds[best_f1_score_index]

        unique_thresholds, avg_warning_times, std_warning_times = self.warning_time_vs_threshold(horizon)
        # Find index where threshold is equal to the best f1 score threshold
        
        if self.alarm_type == 'sthr':
            warning_time_index = np.where(unique_thresholds == best_f1_threshold)
        else:
            # TODO make better
            warning_time_index = -1
            for i in range(len(unique_thresholds)):
                unique_threshold_first = unique_thresholds[i][0]
                unique_threshold_second = unique_thresholds[i][1]
                unique_threshold_third = unique_thresholds[i][2]
                if unique_threshold_first == best_f1_threshold[0] and unique_threshold_second == best_f1_threshold[1] and unique_threshold_third == best_f1_threshold[2]:
                    warning_time_index = i
                    break

        avg_warning_time = avg_warning_times[warning_time_index]
        std_warning_time = std_warning_times[warning_time_index]

        return true_alarm_rate, false_alarm_rate, avg_warning_time, std_warning_time