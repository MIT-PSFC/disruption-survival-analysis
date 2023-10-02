# Class that holds onto data shared between multiple experiments

import numpy as np

from sklearn.metrics import roc_auc_score
from disruption_survival_analysis.manage_datasets import load_dataset
from disruption_survival_analysis.experiment_utils import label_shot_data, calculate_alarm_times, calculate_alarm_times_hysteresis, calculate_alarm_times_ettd, timeslice_micro_avg, area_under_curve, calculate_f1_scores, expected_time_to_disruption_integral, unique_domain_mapping
from disruption_survival_analysis.experiment_utils import SIMPLE_THRESHOLDS
from disruption_survival_analysis.model_utils import get_model_for_experiment, name_model

from auton_survival.estimators import SurvivalModel # CPH, DCPH, DSM, DCM, RSF
from sklearn.ensemble import RandomForestClassifier

from disruption_survival_analysis.DisruptionPredictors import DisruptionPredictorSM, DisruptionPredictorRF, DisruptionPredictorKM

from disruption_survival_analysis.critical_metrics import compute_metrics_vs_thresholds, compute_metrics_vs_false_alarm_rates

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
        self.device = config['device']
        self.dataset_path = config['dataset_path']

        # Set the type of experiment
        self.experiment_type = experiment_type
        # Load data for experiment: all data, including shot, time, time_until_disrupt, and features fed to predictor
        self.all_data = load_dataset(self.device, self.dataset_path, experiment_type)

        # Create the model and predictor for the experiment
        model = get_model_for_experiment(config, experiment_type)
        required_warning_time = config['required_warning_time']
        self.name = name_model(config)

        model_type = config['model_type']

        hyperparameters = config['hyperparameters']

        if model_type in ['cph', 'dcph', 'dsm'] and isinstance(model, SurvivalModel):
            self.predictor = DisruptionPredictorSM(self.name, model, required_warning_time, hyperparameters['horizon'])
        elif model_type in ['rf'] and isinstance(model, RandomForestClassifier):
            self.predictor = DisruptionPredictorRF(self.name, model, required_warning_time, hyperparameters['class_time'])
        elif model_type in ['km'] and isinstance(model, RandomForestClassifier):
            self.predictor = DisruptionPredictorKM(self.name, model, required_warning_time, hyperparameters['class_time'], hyperparameters['fit_time'])
        else:
            raise ValueError('Model type not recognized')
        
        # Get the alarm type from the config
        self.alarm_type = config['alarm_type']

        # Set the thresholds for usage in tpr/fpr calculations
        if self.alarm_type == 'sthr':
            # Simple Threshold
            self.thresholds = SIMPLE_THRESHOLDS
        elif self.alarm_type == 'athr':
            # Uses all unique values of risk for all shots in the dataset at a given horizon
            # This is calculated in the get_all_thresholds method when the metric is being evaluated, since it depends on the horizon
            self.thresholds = None
        elif self.alarm_type == 'hyst':
            # Hysteresis
            # Make list of tuples of (min, max, time) for hysteresis
            # Match the scheme use in Montes 2021
            # min goes from 0.05 to 0.5 in increments of 0.05
            # where max goes from min to 0.95 in increments of 0.05
            # and trigger time goes from 0 to 50 ms in increments of 5 ms
            self.thresholds = []
            for min_threshold in np.linspace(0, 0.5, 11):
                for max_threshold in np.arange(min_threshold, 1, 0.05):
                    for time_threshold in np.linspace(0, 0.05, 11):
                        self.thresholds.append((min_threshold, max_threshold, time_threshold))
        elif self.alarm_type == 'ettd':
            # Expected time to disruption
            self.thresholds = [0.1, 0.05, 0.02] # Expected time to disruption thresholds in seconds
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
    
    def get_disruption_time(self, shot):
        """ Returns the disruption time for a given shot """
        if shot not in self.get_disruptive_shot_list():
            raise ValueError('Shot was not disruptive, cannot get disruption time.')
        else:
            shot_data = self.all_data[self.all_data['shot'] == shot]
            disruption_time = shot_data['time'].iloc[0] + shot_data['time_until_disrupt'].iloc[0]
            return disruption_time
    
    def get_shot_data(self, shot):
        """ Returns the data for a given shot """
        return self.all_data[self.all_data['shot'] == shot]

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

    def get_all_thresholds(self, horizon):
        """ Gets all unique values of risk for all shots in the dataset at a given horizon """
        
        shot_list = self.get_shot_list()

        all_risks = []

        # Iterate through shots
        for i, shot in enumerate(shot_list):
            # Get the alarm times given by the model
            risk_at_times = self.get_predictor_risk_at_times(shot, horizon)
            all_risks.append(risk_at_times['risk'].unique())

        # Get the unique risk values
        unique_risk_values = np.unique(np.concatenate(all_risks))
        # Sort the unique risk values
        unique_risk_values = np.sort(unique_risk_values)

        return unique_risk_values

    # Get info from the predictor

    def get_predictor_risk(self, shot, horizon=None):
        data = self.get_shot_data(shot)
        return self.predictor.get_risk(shot, data, horizon=horizon)
    
    def get_predictor_risk_at_times(self, shot, horizon=None):
        return self.predictor.get_risk_at_times(shot, self.get_shot_data(shot), horizon=horizon)
    
    def get_predictor_ettd(self, shot):
        return self.predictor.get_ettd(shot, self.get_shot_data(shot))
    
    def get_predictor_ettd_at_times(self, shot):
        return self.predictor.get_ettd_at_times(shot, self.get_shot_data(shot))

    # Area Under ROC on a timeslice basis

    def auroc_timeslice_shot(self, horizons, shot):
        """ Calculates the area under ROC curve evaluated on a timeslice basis for a single shot.
        Only defined for disruptive shots, raises ValueError if shot is not disruptive.

        Parameters
        ----------
        horizons : list of floats
            List of horizons in seconds to calculate ROC AUC for
        shot : int
            Shot number to calculate ROC AUC for
        
        Returns
        -------
        auroc_list : list of floats
            List of ROC AUCs corresponding to each horizon
        """

        # Determine if shot was disruptive
        disruptive = shot in self.get_disruptive_shot_list()
        if not disruptive:
            raise ValueError('Shot was not disruptive, cannot calculate ROC AUC because only one class present in y_true.')

        # Get the features for the shot
        shot_data = self.get_shot_data(shot)
        
        auroc_list = []   # List of ROC AUCs corresponding to each horizon
        for horizon in horizons:
            # Set up true labels and predicted risk scores
            y_true = label_shot_data(shot_data, disruptive, horizon)
            y_pred = self.get_predictor_risk(shot, horizon=horizon)

            auroc_list.append(roc_auc_score(y_true, y_pred))

        return auroc_list
    
    def auroc_timeslice_shot_avg(self, horizons):
        """ Calculates the area under ROC curve evaluated on a timeslice basis
        for each individual shot, averaged over all shots.
        Only defined for disruptive shots, since this metric needs more than one class
        in the 'truth' array, and non-disruptive shots only have one class.
        
        Parameters:
        -----------
        horizons : list
            List of horizons to evaluate ROC AUC at
        
        Returns:
        --------
        avg_auroc_array : array
            Array of average ROC AUCs for each horizon
        std_auroc_array : array
            Array of standard deviations of ROC AUCs for each horizon
        """
    
        # Get a list of all disruptive shots in the dataset
        shot_list = self.get_disruptive_shot_list()

        # Iterate through all shots and calculate the ROC AUC for each
        auroc_array = np.zeros((len(shot_list), len(horizons)))
        for i, shot in enumerate(shot_list):
            auroc_array[i,:] = self.auroc_timeslice_shot(horizons, shot)

        # Average the ROC AUCs over all shots
        avg_auroc_array = np.mean(auroc_array, axis=0)
        std_auroc_array = np.std(auroc_array, axis=0)
        return avg_auroc_array, std_auroc_array
    
    def auroc_timeslice_all(self, horizons, disrupt_only=True):
        """ Calculates the area under ROC curve evaluated on a timeslice basis over many horizons.

        Parameters:
        -----------
        horizons : list
            List of horizons to evaluate ROC AUC at
        disrupt_only : bool
            If True, only use disruptive shots in the dataset

        Returns:
        --------
        auroc_list : list
            List of ROC AUCs corresponding to each horizon
        """

        # Get list of shots to use
        if disrupt_only:
            shot_list = self.get_disruptive_shot_list()
        else:
            shot_list = self.get_shot_list()

        auroc_list = []   # List of ROC AUCs corresponding to each horizon
        for horizon in horizons:
            y_true = []
            y_pred = []

            for shot in shot_list:
                # Get the features for the shot
                shot_data = self.get_shot_data(shot)
                # Determine if shot was disruptive
                disruptive = shot in self.get_disruptive_shot_list()
                # Set up true labels and predicted risk scores
                new_labels = label_shot_data(shot_data, disruptive, horizon)
                new_predictions = self.get_predictor_risk(shot, horizon=horizon)
                y_true.extend(new_labels)
                y_pred.extend(new_predictions)

            auroc_list.append(roc_auc_score(y_true, y_pred))

        return auroc_list
    
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

        # If using 'athr' alarm, get all unique floats for thresholds
        if self.alarm_type == 'athr':
            self.thresholds = self.get_all_thresholds(horizon)

        # Create array to store the results
        # Array is of shape (num_shots, num_thresholds)
        alarm_times = np.zeros((len(shot_list), len(self.thresholds)))

        # Determine which function to use 
        if self.alarm_type in ['sthr', 'athr']:
            # Simple Threshold

            # Iterate through shots
            for i, shot in enumerate(shot_list):
                # Get the alarm times given by the model
                risk_at_times = self.get_predictor_risk_at_times(shot, horizon)

                alarm_times_calced = calculate_alarm_times(risk_at_times, self.thresholds)

                # Save the alarm times
                alarm_times[i,:] = alarm_times_calced
        
        elif self.alarm_type == 'hyst':
            # Hysteresis

            # Iterate through shots
            for i, shot in enumerate(shot_list):
                # Get the alarm times given by the model
                risk_at_times = self.get_predictor_risk_at_times(shot, horizon)
                alarm_times_calced = calculate_alarm_times_hysteresis(risk_at_times, self.thresholds)

                # Save the alarm times
                alarm_times[i,:] = alarm_times_calced
        
        elif self.alarm_type == 'ettd':
            # Expected Time to Disruption

            # Iterate through shots
            for i, shot in enumerate(shot_list):                
                # Get the expected time to disruption of the shot
                expected_lifetime = self.get_predictor_ettd_at_times(shot)
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

        # Get the alarm times given by the model
        alarm_times = self.get_alarm_times(horizon)

        # Create arrays to store the results
        # Array is of shape (num_shots, num_thresholds)
        true_alarms = np.zeros((len(shot_list), len(self.thresholds)))
        false_alarms = np.zeros((len(shot_list), len(self.thresholds)))

        # Iterate through shots
        for i, shot in enumerate(shot_list):
            disrupt = shot in disruptive_shots

            # If the shot is disruptive, find the time of disruption
            # Even if the shot is non-disruptive, we will determine a warning time for the false alarm calculation
            if disrupt:
                disrupt_time = self.get_disruption_time(shot)
                warning_times = disrupt_time - alarm_times[i,:]
            else:
                final_time = self.get_shot_data(shot)['time'].max()
                warning_times = final_time - alarm_times[i,:]
        
            # Fill in true and false alarms
            # True alarm is when shot is disruptive and warning time is greater than required warning time
            # False alarm is when shot is not disruptive but an alarm time is still given
            true_alarms[i,:] = (disrupt & (warning_times > required_warning_time)).astype(int)
            false_alarms[i,:] = (~disrupt & ~np.isnan(alarm_times[i,:])).astype(int)
            
        return true_alarms, false_alarms
    
    def true_false_alarm_rates(self, horizon=None, required_warning_time=None):
        """ Get statistics on false alarm rate vs threshold for a given horizon and required warning time
            This is inherently a macro statistic, since a single shot can have only one alarm at a given threshold
        """

        true_alarms, false_alarms = self.get_true_false_alarms(horizon, required_warning_time)

        true_alarm_rates = np.sum(true_alarms, axis=0) / self.get_num_disruptive_shots() 
        false_alarm_rates = np.sum(false_alarms, axis=0) / self.get_num_non_disruptive_shots()

        return true_alarm_rates, false_alarm_rates

    def true_alarm_rate_vs_false_alarm_rate(self, horizon=None, required_warning_time=None):

        true_alarm_rates, false_alarm_rates = self.true_false_alarm_rates(horizon, required_warning_time)

        unique_false_alarms, avg_true_alarm_rates, std_true_alarm_rates = unique_domain_mapping(false_alarm_rates, true_alarm_rates)

        return unique_false_alarms, avg_true_alarm_rates#, std_true_alarm_rates
     
    def missed_alarm_rate_vs_false_alarm_rate(self, horizon=None, required_warning_time=None):

        true_alarms, false_alarms = self.get_true_false_alarms(horizon, required_warning_time)

        missed_alarm_rates = 1 - np.sum(true_alarms, axis=0) / self.get_num_disruptive_shots()
        false_alarm_rates = np.sum(false_alarms, axis=0) / self.get_num_non_disruptive_shots()

        unique_false_alarms, avg_missed_alarm_rates, std_missed_alarm_rates = unique_domain_mapping(false_alarm_rates, missed_alarm_rates)

        return unique_false_alarms, avg_missed_alarm_rates
    
    def false_alarm_rate_vs_threshold(self, required_warning_time=None, horizon=None, method='average'):

        _, false_alarm_rates = self.true_false_alarm_rates(horizon, required_warning_time)

        unique_thresholds, typical_warning_times, spread_warning_times = unique_domain_mapping(self.thresholds, false_alarm_rates, method=method)

        return unique_thresholds, typical_warning_times, spread_warning_times

    # Warning Times Methods

    def get_warning_times_list(self, horizon=None, required_warning_time=None):
        """Get a list of warning times for each threshold given a horizon.
        This is a list of arrays of variable length,
        The list is ordered to be consistent with the list of thresholds.

        Parameters
        ----------
        horizon : float, optional
            The horizon to get warning times for
            If None, use the horizon the model was hyperparameter tuned for
        required_warning_time : float, optional
            The required warning time. If None, return all warning times
            If specified, only return warning times greater than the required warning time
        """
        
        # Get list of all shots
        shot_list = self.get_shot_list()
        # Get list of disruptive shots
        disruptive_shots = self.get_disruptive_shot_list()

        # Get array of alarm times for this horizon
        alarm_times = self.get_alarm_times(horizon)

        # Create list to store warning times for each shot at various thresholds
        warning_times_shot_list = []
        for disruptive_shot in disruptive_shots:
            # Get the time of disruption for this shot
            disrupt_time = self.get_disruption_time(disruptive_shot)
            # Find the index of the shot in the shot list
            shot_index = np.where(shot_list == disruptive_shot)[0][0]
            # Get the alarm times for this shot
            shot_alarm_times = alarm_times[shot_index]

            # Calculate the warning times for this shot
            warning_times = [disrupt_time - alarm_time for alarm_time in shot_alarm_times]
            
            # If warning time is nan, set it to zero
            # NOTE: if warning time is None, that means no alarm was raised.
            # Should this be treated as a zero warning time, or should it be ignored?
            # It could go either way. We have chosen that it should be treated as a zero warning time.
            # This makes the 'average warning time vs false positve rate plot' more intuitive,
            # as the average warning time is then monatonically increasing with false positive rate.
            # If one were to pick the other option, then the average warning time could 
            # fluctuate up and down with false positive rate
            # (Consider the case where a single disruptive shot has a moderate risk value very early on,
            # and many other disruptive shots have a lower risk value very close to the end)
            warning_times = [warning_time if warning_time >= 0 else 0 for warning_time in warning_times]

            warning_times_shot_list.append(warning_times)

        # The statistics we want to do are for each threshold for various shots
        # So we need to transpose the list

        # First, turn into a numpy array
        warning_times_shot_list = np.array(warning_times_shot_list)
        # Then, transpose
        warning_times_threshold_list = warning_times_shot_list.T

        # If there is a required warning time, remove warning times less than that
        # TODO: clarify that warning time does not take into account the cutoff,
        # but true alarms and false alarms *do*
        #if required_warning_time != None:
        #    warning_times_threshold_list = [[warning_time for warning_time in warning_times if warning_time > required_warning_time] for warning_times in warning_times_threshold_list]

        return warning_times_threshold_list

    def warning_time_vs_threshold(self, horizon=None, required_warning_time=None, method='average'):
        """ Get statistics on warning time vs threshold for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one warning time
        """

        warning_times_list = self.get_warning_times_list(horizon, required_warning_time)

        unique_thresholds, typical_warning_times, spread_warning_times = unique_domain_mapping(self.thresholds, warning_times_list, method=method)

        return unique_thresholds, typical_warning_times, spread_warning_times

    def warning_time_vs_false_alarm_rate(self, horizon=None, required_warning_time=None, method='average'):
        """ Get statistics on warning time vs false alarm rate for a given horizon 
            This is inherently a macro statistic, since a single shot can have only one warning time
        """

        if horizon == None:
            horizon = self.predictor.trained_disruptive_window
        if required_warning_time == None:
            required_warning_time = self.predictor.trained_required_warning_time

        # Ger warning time list filtered by required warning time
        warning_times_list = self.get_warning_times_list(horizon, required_warning_time=required_warning_time)
        
        _, false_alarm_rates = self.true_false_alarm_rates(horizon, required_warning_time)

        unique_false_alarm_rates, typical_warning_times, spread_warning_times = unique_domain_mapping(false_alarm_rates, warning_times_list, method=method)

        # TODO: Ignore zero false positve rate results???
        return unique_false_alarm_rates, typical_warning_times, spread_warning_times
    
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
            metric_val = timeslice_micro_avg(self.device, self.dataset_path, self.predictor.model, self.experiment_type)
        elif metric_type == 'auroc':
            # Area under ROC curve
            false_alarm_rates, true_alarm_rates = self.true_alarm_rate_vs_false_alarm_rate(horizon, required_warning_time)
            metric_val = area_under_curve(false_alarm_rates, true_alarm_rates)
        elif metric_type == 'auwtc':
            # Area under warning time curve
            false_alarm_rates, warning_times, _ = self.warning_time_vs_false_alarm_rate(horizon, required_warning_time, method='average')
            metric_val = area_under_curve(false_alarm_rates, warning_times, x_cutoff=0.05)
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
        false_alarm_rate = false_alarm_count_array[best_f1_score_index]/(self.get_num_non_disruptive_shots())
        
        # Get threshold at the best f1 score
        best_f1_threshold = self.thresholds[best_f1_score_index]

        unique_thresholds, avg_warning_times, std_warning_times = self.warning_time_vs_threshold(horizon)
        # Find index where threshold is equal to the best f1 score threshold
        
        if self.alarm_type != 'hyst':
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
    
    def critical_metric_setup(self, horizon):

        if self.alarm_type not in ['sthr', 'athr']:
            raise ValueError('Critical metric only defined for simple threshold alarms (sthr, athr)')

        # 0. Set up the predictions and outcomes to calculate the metric

        # Group data by shot
        shot_data_list = self.all_data.groupby('shot')
        # Sort data by time
        shot_data_list = [shot_data.sort_values('time') for _, shot_data in shot_data_list]

        # Find the features in the data
        feature_names = list(self.all_data.columns)
        # Remove the 'shot', 'time', and 'time_until_disrupt' columns
        feature_names.remove('shot')
        feature_names.remove('time')
        feature_names.remove('time_until_disrupt')

        predictions = []
        true_outcomes = []
        for shot_data in shot_data_list:
            shot_predictions = {}
            feature_data = shot_data[feature_names]
            # Predict risk depending on the model type
            if isinstance(self.predictor.model, SurvivalModel):
                try:
                    shot_predictions['risk'] = self.predictor.model.predict_risk(feature_data, horizon)
                except:
                    # DSM expects horizon in a list
                    shot_predictions['risk'] = self.predictor.model.predict_risk(feature_data, [horizon])
            elif isinstance(self.predictor.model, RandomForestClassifier):
                shot_predictions['risk'] = self.predictor.model.predict_proba(feature_data)[:,1]
            else:
                raise ValueError('Model type not supported')
            shot_predictions['time'] = shot_data['time'].values
            # Convert to numpy array
            shot_predictions['risk'] = np.array(shot_predictions['risk'])
            predictions.append(shot_predictions)

            # Determine the time of disruption and whether or not the shot actually disrupted
            true_outcome = {}
            if (shot_data['time_until_disrupt'] >= 0).any():
                true_outcome['disruption_time'] = self.get_disruption_time(shot_data['shot'].iloc[0])
                true_outcome['disrupted'] = True
            elif (shot_data['time_until_disrupt'].isnull()).all():
                true_outcome['disruption_time'] = np.nan
                true_outcome['disrupted'] = False
            else:
                raise ValueError('Invalid shot data, mixed disruption and non-disruption data')
            true_outcomes.append(true_outcome)

        return predictions, true_outcomes


    def get_critical_metrics_vs_thresholds(self, horizon, required_warning_time):

        predictions, outcomes = self.critical_metric_setup(horizon)

        # NOW: Find the false positive rate and average warning times
        self.thresholds = self.get_all_thresholds(horizon)
        true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times = compute_metrics_vs_thresholds(predictions, outcomes, required_warning_time, self.thresholds)

        # 4. Return the false alarm rates and average warning times
        return true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times


    # A self-contained single function to evaluate the only metric which is absolutely critical to the project
    def get_critical_metrics_vs_false_alarm_rate(self, horizon, required_warning_time):
        """ Get the critical metric for the experiment
        This metric is the average warning time (y-axis, range) vs false positive rate (x-axis, domain)
        Only applicable to simple threshold alarms
        
        Parameters
        ----------
        horizon : float
            The horizon for the survival models to predict at
        required_warning_time : float
            The time before a disruption an alarm must be triggered for it to count as a 'true alarm'

        Returns
        -------
        unique_false_alarm_rates, avg_warning_times, std_warning_times : np.array, np.array, np.array
            The unique false alarm rates, average warning times, and standard deviation of warning times
        """

        if self.alarm_type not in ['sthr', 'athr']:
            raise ValueError('Critical metric only defined for simple threshold alarms (sthr, athr)')

        predictions, true_outcomes = self.critical_metric_setup(horizon)


        unique_false_alarm_rates, avg_warning_times, std_warning_times = compute_metrics_vs_thresholds_parallel(predictions, true_outcomes, required_warning_time, self.thresholds)

        # 4. Return the false alarm rates and average warning times
        return unique_false_alarm_rates, avg_warning_times, std_warning_times




        
