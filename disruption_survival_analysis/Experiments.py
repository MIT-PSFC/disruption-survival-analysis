# Class that holds onto data shared between multiple experiments

import numpy as np

from sklearn.metrics import roc_auc_score
from disruption_survival_analysis.manage_datasets import load_dataset
from disruption_survival_analysis.experiment_utils import label_shot_data, timeslice_micro_avg, area_under_curve, calculate_f1_scores
from disruption_survival_analysis.model_utils import get_model_for_experiment, name_model

from auton_survival.estimators import SurvivalModel # CPH, DCPH, DCM, DSM, RSF
from sklearn.ensemble import RandomForestClassifier # RF, KM

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

        if model_type in ['cph', 'dcph', 'dcm', 'dsm'] and isinstance(model, SurvivalModel):
            self.predictor = DisruptionPredictorSM(self.name, model, required_warning_time, hyperparameters['horizon'])
        elif model_type in ['rf'] and isinstance(model, RandomForestClassifier):
            self.predictor = DisruptionPredictorRF(self.name, model, required_warning_time, hyperparameters['class_time'])
        elif model_type in ['km'] and isinstance(model, RandomForestClassifier):
            self.predictor = DisruptionPredictorKM(self.name, model, required_warning_time, hyperparameters['class_time'], hyperparameters['fit_time'])
        else:
            raise ValueError('Model type not recognized')
        
        # Get the alarm type from the config
        self.alarm_type = config['alarm_type']

        # For now, all using 'sthr' alarm type
        self.thresholds = None

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
    
    # Vs. Thresholds Metrics
        
    def true_alarm_rates_vs_thresholds(self, horizon=None, required_warning_time=None):
        """
        Returns:
        --------
        thresholds : np.array
            Every unique float returned as a risk by the model for the entire dataset, to be use as alarm thresholds
        true_alarm_rates : np.array
            True alarm rates corresponding to each threshold
        """
        thresholds, true_alarm_rates, _, _, _ = self.get_critical_metrics_vs_thresholds(horizon, required_warning_time)
        return thresholds, true_alarm_rates

    def false_alarm_rates_vs_thresholds(self, horizon=None, required_warning_time=None):
        """
        Returns:
        --------
        thresholds : np.array
            Every unique float returned as a risk by the model for the entire dataset, to be use as alarm thresholds
        false_alarm_rates : np.array
            False alarm rates corresponding to each threshold
        """
        thresholds, _, false_alarm_rates, _, _ = self.get_critical_metrics_vs_thresholds(horizon, required_warning_time)
        return thresholds, false_alarm_rates
    
    def warning_times_vs_thresholds(self, required_warning_time=None, horizon=None):
        """
        Returns:
        --------
        thresholds : np.array
            Every unique float returned as a risk by the model for the entire dataset, to be use as alarm thresholds
        avg_warning_times : np.array
            Average warning times corresponding to each threshold
        std_warning_times : np.array
            Standard deviations of warning times corresponding to each threshold
        """
        thresholds, _, _, avg_warning_times, std_warning_times = self.get_critical_metrics_vs_thresholds(horizon, required_warning_time)
        return thresholds, avg_warning_times, std_warning_times
    
    # Vs. False Alarm Rate Metrics

    def true_alarm_rates_vs_false_alarm_rates(self, horizon=None, required_warning_time=None):
        """
        Returns:
        --------
        unique_false_alarm_rates : np.array
            Every unique false alarm rate for the entire dataset
        true_alarm_rates : np.array
            Average true alarm rates corresponding to each false alarm rate
        """
        unique_false_alarm_rates, true_alarm_rates, _, _ = self.get_critical_metrics_vs_false_alarm_rates(horizon, required_warning_time)
        return unique_false_alarm_rates, true_alarm_rates
    
    def warning_times_vs_false_alarm_rates(self, horizon=None, required_warning_time=None):
        """
        Returns:
        --------
        unique_false_alarm_rates : np.array
            Every unique false alarm rate for the entire dataset
        avg_warning_times : np.array
            Average warning times corresponding to each false alarm rate
        std_warning_times : np.array
            Standard deviations of warning times corresponding to each false alarm rate
        """
        unique_false_alarm_rates, _, avg_warning_times, std_warning_times = self.get_critical_metrics_vs_false_alarm_rates(horizon, required_warning_time)
        return unique_false_alarm_rates, avg_warning_times, std_warning_times
    
    # Metrics methods

    def evaluate_metric(self, metric_type, horizon=None, required_warning_time=None):
        """
        Evaluate a metric on the experiment

        Parameters
        ----------
        metric_type : str
            The type of metric to evaluate
            Options are 'tslic', 'auroc', 'auwtc', 'maxf1'
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
            false_alarm_rates, true_alarm_rates = self.true_alarm_rates_vs_false_alarm_rates(horizon, required_warning_time)
            metric_val = area_under_curve(false_alarm_rates, true_alarm_rates)
        elif metric_type == 'auwtc':
            # Area under warning time curve
            false_alarm_rates, warning_times, _ = self.warning_times_vs_false_alarm_rates(horizon, required_warning_time)
            metric_val = area_under_curve(false_alarm_rates, warning_times, x_cutoff=0.05)
        elif metric_type == 'maxf1':
            # Highest f1 score over all the thresholds
            _, true_alarm_rates, false_alarm_rates, _, _ = self.get_critical_metrics_vs_thresholds(horizon, required_warning_time)

            metric_val = None
        else:
            metric_val = None

        return metric_val
    
    def critical_metric_setup(self, horizon):
        """ Set up data for calculating critical metrics 
        
        Parameters
        ----------
        horizon : float
            The horizon for the survival models to predict at
        
        Returns
        -------
        unique_thresholds: np.array
            List of every unique float returned as a risk by the model for the entire dataset, to be use as alarm thresholds
        predictions : list of dicts
            List of dictionaries containing the following keys:
            'risk' : np.array
                Array of predicted risks for the shot
            'time' : np.array
                Array of times for the shot
        outcomes : list of dicts
            List of dictionaries containing the following keys:
            'disruption_time' : float
                The time of disruption for the shot
            'disrupted' : bool
                Whether or not the shot disrupted
        """

        # Group data by shot and sort by time
        shot_data_list = self.all_data.groupby('shot')
        shot_data_list = [shot_data.sort_values('time') for _, shot_data in shot_data_list]

        all_thresholds = np.array([0, 1]) # Ensure that 0 and 1 are included in the list of thresholds
        predictions = []
        outcomes = []
        for shot_data in shot_data_list:
            # Predict risk depending on the model type
            shot_predictions = {}
            if isinstance(self.predictor.model, SurvivalModel):
                shot_predictions['risk'] = self.predictor.get_risks(shot_data, horizon)
            elif isinstance(self.predictor, DisruptionPredictorRF):
                shot_predictions['risk'] = self.predictor.get_risks(shot_data)
            elif isinstance(self.predictor, DisruptionPredictorKM):
                shot_predictions['risk'] = self.predictor.get_risks(shot_data, horizon)
            else:
                raise ValueError('Model type not supported')
            
            # Add the thresholds to the list of all thresholds
            all_thresholds = np.concatenate((all_thresholds, shot_predictions['risk']))

            # Add prediction for this shot
            shot_predictions['time'] = shot_data['time'].values
            predictions.append(shot_predictions)

            # Determine the time of disruption and whether or not the shot actually disrupted
            outcome = {}
            if (shot_data['time_until_disrupt'] >= 0).any():
                outcome['disruption_time'] = self.get_disruption_time(shot_data['shot'].iloc[0])
                outcome['disrupted'] = True
            elif (shot_data['time_until_disrupt'].isnull()).all():
                outcome['disruption_time'] = np.nan
                outcome['disrupted'] = False
            else:
                raise ValueError('Invalid shot data, mixed disruption and non-disruption data')
            outcomes.append(outcome)

        unique_thresholds = np.unique(all_thresholds)

        return unique_thresholds, predictions, outcomes

    def get_critical_metrics_vs_thresholds(self, horizon=None, required_warning_time=None):
        """ Get critical metrics for the experiment, where each metric is compared at alarm thresholds
        
        Parameters
        ----------
        horizon : float. optional
            The horizon for the survival models to predict at. If None, use the horizon the model was hyperparameter tuned for
        required_warning_time : float, optional
            The required warning time to evaluate the metric at. If None, use the required warning time the model was trained on
        
        Returns
        -------
        thresholds : np.array
            Every unique float returned as a risk by the model for the entire dataset, to be used as alarm thresholds
        true_alarm_rates : np.array
            True alarm rates corresponding to each threshold
        false_alarm_rates : np.array
            False alarm rates corresponding to each threshold
        avg_warning_times : np.array
            Average warning times corresponding to each threshold
        std_warning_times : np.array
            Standard deviations of warning times corresponding to each threshold
        """

        if horizon is None:
            horizon = self.predictor.horizon
        if required_warning_time is None:
            required_warning_time = self.required_warning_time

        thresholds, predictions, outcomes = self.critical_metric_setup(horizon)

        true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times = compute_metrics_vs_thresholds(predictions, outcomes, required_warning_time, thresholds)

        return thresholds, true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times

    def get_critical_metrics_vs_false_alarm_rates(self, horizon=None, required_warning_time=None):
        """ Get critical metrics for the experiment, where each metric is compared with false alarm rates

        Parameters
        ----------
        horizon : float. optional
            The horizon for the survival models to predict at. If None, use the horizon the model was hyperparameter tuned for
        required_warning_time : float, optional
            The required warning time to evaluate the metric at. If None, use the required warning time the model was trained on
        
        Returns
        -------
        unique_false_alarm_rates : np.array
            Every unique false alarm rate for the entire dataset
        true_alarm_rates : np.array
            Average true alarm rates corresponding to each false alarm rate
        avg_warning_times : np.array
            Average warning times corresponding to each false alarm rate
        std_warning_times : np.array
            Standard deviations of warning times corresponding to each false alarm rate
        """

        if horizon is None:
            try:
                horizon = self.predictor.trained_horizon
            except:
                # If horizon is not defined, in the case of a binary classifier, just pass
                pass
        if required_warning_time is None:
            required_warning_time = self.predictor.trained_required_warning_time

        thresholds, predictions, outcomes = self.critical_metric_setup(horizon)

        unique_false_alarm_rates, true_alarm_rates, avg_warning_times, std_warning_times = compute_metrics_vs_false_alarm_rates(predictions, outcomes, required_warning_time, thresholds)

        return unique_false_alarm_rates, true_alarm_rates, avg_warning_times, std_warning_times
