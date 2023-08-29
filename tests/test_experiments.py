"""Test functions to ensure that the data preprocessing works as expected
"""
import unittest

from disruption_survival_analysis.Experiments import Experiment
from disruption_survival_analysis.experiment_utils import load_experiment_config
from disruption_survival_analysis.manage_datasets import load_disruptive_shot_list

#from disruption_survival_analysis.model_utils import load_model

TEST_DEVICE = 'synthetic'
TEST_DATASET_PATH = 'synthetic100'

class TestSimpleFunctions(unittest.TestCase):

    def setUp(self):
        """Set up the test case
        """

        # Load the config for a simple DSM experiment
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'dsm', 'sthr', 'auroc', 0.02)
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

class TestAllCombos(unittest.TestCase):

    # Test for every combination of model, alarm type, and metric
    model_list = ['cph', 'dcph', 'dsm', 'rf', 'km']
    alarm_type_list = ['sthr', 'hyst']
    metric_list = ['auroc', 'auwtc']
    min_required_warning_times = [0.02]

    def test_evaluate_all_metrics(self):
        """Test that all metrics can be evaluated for all experiments on the validation set"""

        # Load all experiments of the various types
        self.experiments = []
        for model in self.model_list:
            for alarm_type in self.alarm_type_list:
                for metric in self.metric_list:
                    for min_required_warning_time in self.min_required_warning_times:
                        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, model, alarm_type, metric, min_required_warning_time)
                        self.experiments.append(Experiment(experiment_config, 'test'))

        # Evaluate the metrics for each experiment
        for experiment in self.experiments:
            for metric in self.metric_list:
                try:
                    experiment.evaluate_metric(metric)
                except:
                    self.fail(f"Failed to evaluate metric {metric} for experiment {experiment.name}")

    def test_evaluate_single_metric(self):
        """Used to test a single metric for a single experiment"""

        test_metric = 'auwtc'
        
        model = 'cph'
        alarm_type = 'sthr'
        model_metric = 'auroc'
        min_required_warning_time = 0.02
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, model, alarm_type, model_metric, min_required_warning_time)
        experiment = Experiment(experiment_config, 'test')

        experiment.evaluate_metric(test_metric)


class TestCriticalMetric(unittest.TestCase):

    def setUp(self):
        """Set up the test case
        """

        # Load simple DSM experiment
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'dsm', 'sthr', 'auroc', 0.02)
        self.experiment = Experiment(experiment_config, 'test')

    def test_get_metric(self):
        """Test that the metric is obtained correctly
        """

        epsilon = 0.001

        # Get the metric both ways
        direct_false_alarm_rates, direct_avg_warning_times, direct_std_warning_times = self.experiment.get_critical_metric(horizon=0.05, required_warning_time=0.02)
        general_false_alarm_rates, general_avg_warning_times, general_std_warning_times = self.experiment.warning_time_vs_false_alarm_rate(horizon=0.05, required_warning_time=0.02)

        # Check that the metric is calculated correctly, within epsilon
        if (abs(direct_false_alarm_rates - general_false_alarm_rates) > epsilon).any():
            self.fail("False alarm rates are not equal")
        for i in range(len(direct_false_alarm_rates)):
            if (abs(direct_avg_warning_times[i] - general_avg_warning_times[i]) > epsilon).any():
                self.fail("Average warning times are not equal")
            if (abs(direct_std_warning_times[i] - general_std_warning_times[i]) > epsilon).any():
                self.fail("Standard deviation of warning times are not equal")


class TestWarningTimesList(unittest.TestCase):

    def setUp(self):
        """Set up the test case
        """

        # Load simple DSM experiment
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'dsm', 'sthr', 'auroc', 0.02)
        self.experiment = Experiment(experiment_config, 'test')

    def test_no_negative_warning_times(self):
        """Ensure that there are no negative warning times
        """

        # Get the warning times list
        warning_times_list = self.experiment.get_warning_times_list(horizon=0.05)

        # Check that there are no negative warning times
        for warning_times in warning_times_list:
            for warning_time in warning_times:
                self.assertGreaterEqual(warning_time, 0)


