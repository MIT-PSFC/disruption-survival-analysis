"""Test functions to ensure that the data preprocessing works as expected
"""
import unittest
import numpy as np

from tests.test_manage_datasets import TEST_DEVICE, TEST_DATASET_PATH

from disruption_survival_analysis.Experiments import Experiment
from disruption_survival_analysis.experiment_utils import load_experiment_config
from disruption_survival_analysis.manage_datasets import load_disruptive_shot_list

class TestSimpleFunctions(unittest.TestCase):

    def setUp(self):
        """Set up the test case
        """
        # Load the config for a simple RF experiment
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'rf', 'sthr', 'auroc', 0.01)
        self.experiment = Experiment(experiment_config, 'test')
        
    def test_get_num_disrupt(self):
        """Test that the number of disruptions is calculated correctly
        """

        # Get the number of disruptions in the test set
        true_disruptive_shot_count = len(load_disruptive_shot_list(TEST_DEVICE, TEST_DATASET_PATH, 'test'))

        # Calculate the number of disruptions in the test set
        get_num_disruptive_shots_result = self.experiment.get_num_disruptive_shots()

        self.assertEqual(true_disruptive_shot_count, get_num_disruptive_shots_result)

    def test_consistent_shot_ordering(self):
        """Ensure that disruptive and non-disruptive shot lists are ordered consistently
        with the full shot list
        """

        # Get each shot list
        all_shots = self.experiment.get_shot_list()
        disruptive_shots = self.experiment.get_disruptive_shot_list()
        non_disruptive_shots = self.experiment.get_non_disruptive_shot_list()

        # Check that the shot lists are ordered consistently
        next_disruptive_shot = 0
        next_non_disruptive_shot = 0
        for shot in all_shots:
            if next_disruptive_shot < len(disruptive_shots) and shot == disruptive_shots[next_disruptive_shot]:
                    next_disruptive_shot += 1
            elif next_non_disruptive_shot < len(non_disruptive_shots) and next_non_disruptive_shot < len(non_disruptive_shots):
                if shot == non_disruptive_shots[next_non_disruptive_shot]:
                    next_non_disruptive_shot += 1
            else:
                self.fail("Shot lists are not ordered consistently")

# class TestCriticalMetricsVsThresholds(unittest.TestCase):

#     def setUp(self):
#         # Load the config for a simple RF experiment
#         experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'rf', 'sthr', 'auroc', 0.01)
#         self.experiment = Experiment(experiment_config, 'test')

#         # Get the metrics
#         self.thresholds, self.true_alarm_rates, self.false_alarm_rates, self.avg_warning_times, self.std_warning_times = self.experiment.get_critical_metrics_vs_thresholds()

#     def test_alarm_rates_boundaries(self):
#         """Check that the alarm rates are between 0 and 1
#         """
#         if np.any(self.true_alarm_rates < 0) or np.any(self.true_alarm_rates > 1):
#             self.fail("True alarm rates are not between 0 and 1")
#         if np.any(self.false_alarm_rates < 0) or np.any(self.false_alarm_rates > 1):
#             self.fail("False alarm rates are not between 0 and 1")

#         # Check that the alarm rates are 1 for a threshold of 0
#         if self.true_alarm_rates[0] != 1:
#             self.fail("True alarm rate is not 1 for threshold of 0")
#         if self.false_alarm_rates[0] != 1:
#             self.fail("False alarm rate is not 1 for threshold of 0")

#         # Check that the alarm rates are 0 for a threshold of 1
#         if self.true_alarm_rates[-1] != 0:
#             self.fail("True alarm rate is not 0 for threshold of 1")
#         if self.false_alarm_rates[-1] != 0:
#             self.fail("False alarm rate is not 0 for threshold of 1")

#     def test_consistent_ordering(self):
#         """Check that the alarm rates and warning times are decreasing while threshold is increasing
#         """
#         for i in range(len(self.thresholds)-1):
#             if self.thresholds[i+1] < self.thresholds[i]:
#                 self.fail("Thresholds are not increasing")
#             if self.true_alarm_rates[i+1] > self.true_alarm_rates[i]:
#                 self.fail("True alarm rates are not decreasing")
#             if self.false_alarm_rates[i+1] > self.false_alarm_rates[i]:
#                 self.fail("False alarm rates are not decreasing")
#             if self.avg_warning_times[i+1] > self.avg_warning_times[i]:
#                 self.fail("Average warning times are not decreasing")

#     def test_warning_times_true_alarm_rates_agree(self):
#         """For every threshold where the true alarm rate is greater than 0, 
#         the average warning time should also be greater than 0"""

#         for i in range(len(self.thresholds)):
#             if self.true_alarm_rates[i] > 0 and self.avg_warning_times[i] == 0:
#                 self.fail("Average warning time is 0 for a threshold where the true alarm rate is greater than 0")
#             if self.avg_warning_times[i] > 0 and self.true_alarm_rates[i] == 0:
#                 self.fail("True alarm rate is 0 for a threshold where the average warning time is greater than 0")

class TestCriticalMetricsVsFalseAlarmRates(unittest.TestCase):

    def setUp(self):
        # Load the config for a simple RF experiment
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'rf', 'sthr', 'auroc', 0.01)
        self.experiment = Experiment(experiment_config, 'test')

        # Get the metrics
        self.unique_false_alarm_rates, self.true_alarm_metrics, self.warning_time_metrics = self.experiment.get_critical_metrics_vs_false_alarm_rates()

        self.true_alarm_rates = self.true_alarm_metrics['avg']
        self.avg_warning_times = self.warning_time_metrics['avg']

    def test_alarm_rates_boundaries(self):
        """Check that the alarm rates are between 0 and 1
        """

        if np.any(self.unique_false_alarm_rates < 0) or np.any(self.unique_false_alarm_rates > 1):
            self.fail("False alarm rates are not between 0 and 1")
        if np.any(self.true_alarm_metrics['avg'] < 0) or np.any(self.true_alarm_metrics['avg'] > 1):
            self.fail("True alarm rates are not between 0 and 1")

        # Check that the minimum false alarm rate is 0 and the maximum is 1
        if self.unique_false_alarm_rates[0] != 0:
            self.fail("Minimum false alarm rate is not 0")
        if self.unique_false_alarm_rates[-1] != 1:
            self.fail("Maximum false alarm rate is not 1")

    def test_consistent_ordering(self):
        """Check that the alarm rates and warning times are increasing while false alarm rate is increasing"
        """

        for i in range(len(self.unique_false_alarm_rates)-1):
            if self.unique_false_alarm_rates[i+1] < self.unique_false_alarm_rates[i]:
                self.fail("False alarm rates are not increasing")
            if self.true_alarm_rates[i+1] < self.true_alarm_rates[i]:
                self.fail("True alarm rates are not increasing")
            if self.avg_warning_times[i+1] < self.avg_warning_times[i]:
                self.fail("Average warning times are not increasing")

    def test_warning_times_true_alarm_rates_agree(self):
        """For every false alarm rate where the true alarm rate is greater than 0, 
        the average warning time should also be greater than 0"""

        for i in range(len(self.unique_false_alarm_rates)):
            if self.true_alarm_rates[i] > 0 and self.avg_warning_times[i] == 0:
                self.fail("Average warning time is 0 for a false alarm rate where the true alarm rate is greater than 0")
            if self.avg_warning_times[i] > 0 and self.true_alarm_rates[i] == 0:
                self.fail("True alarm rate is 0 for a false alarm rate where the average warning time is greater than 0")

class TestAllCombos(unittest.TestCase):
    # Test for every combination of model, alarm type, and metric
    # model_list = ['cph', 'dsm', 'rf', 'km']
    # alarm_type_list = ['sthr']
    # metric_list = ['auroc', 'auwtc']
    # min_required_warning_times = [0.01, 0.05, 0.1, 0.2]
    
    model_list = ['cph', 'dsm', 'rf', 'km']
    alarm_type_list = ['sthr']
    metric_list = ['auwtc']
    min_required_warning_times = [0.01]

    def test_evaluate_all_metrics(self):
        """Test that all metrics can be evaluated for all experiments on the test and validations sets"""

        # Load all experiments of the various types
        self.experiments = []
        for model in self.model_list:
            for alarm_type in self.alarm_type_list:
                for metric in self.metric_list:
                    for min_required_warning_time in self.min_required_warning_times:
                        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, model, alarm_type, metric, min_required_warning_time)
                        self.experiments.append(Experiment(experiment_config, 'test'))
                        self.experiments.append(Experiment(experiment_config, 'val'))

        # Evaluate the metrics for each experiment
        for experiment in self.experiments:
            for metric in self.metric_list:
                try:
                    experiment.evaluate_metric(metric)
                except:
                    self.fail(f"Failed to evaluate metric {metric} for experiment {experiment.name} on the {experiment.experiment_type} set")

    def test_evaluate_single_metric(self):
        """Used to test a single metric for a single experiment"""
        
        model = 'rf'
        alarm_type = 'sthr'
        model_metric = 'auroc'
        min_required_warning_time = 0.01
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, model, alarm_type, model_metric, min_required_warning_time)
        experiment = Experiment(experiment_config, 'test')

        experiment.evaluate_metric(model_metric)

