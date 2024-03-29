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
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'rf', 'sthr', 'auroc', 0.02)
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

class TestTrueAlarmRates(unittest.TestCase):

    def setUp(self):
        # Load simple RF experiment
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'rf', 'sthr', 'auroc', 0.02)
        self.experiment = Experiment(experiment_config, 'test')

    def test_true_alarm_rates_constant_values(self):
        """Ensures that the true alarm rates don't look wrong"""

        # Get the true alarm rates
        true_alarm_rates, _ = self.experiment.true_false_alarm_rates()

        # Check that the true alarm rates are not all 0 or 1
        if (true_alarm_rates == 0).all():
            self.fail("True alarm rates are all 0")
        if (true_alarm_rates == 1).all():
            self.fail("True alarm rates are all 1")

    def test_true_alarm_rates_decreasing(self):
        """Ensures that the true alarm rates are decreasing with higher thresholds"""

        # Get the true alarm rates
        true_alarm_rates, _ = self.experiment.true_false_alarm_rates()

        # Check that the true alarm rates are increasing
        for i in range(len(true_alarm_rates) - 1):
            if true_alarm_rates[i] < true_alarm_rates[i + 1]:
                self.fail("True alarm rates are not decreasing")

    def test_true_alarm_rates_boundaries(self):
        """Check that the true alarm rates are all between 0 and 1"""
            
        # Get the true alarm rates
        true_alarm_rates, _ = self.experiment.true_false_alarm_rates()

        # Check that the true alarm rates are between 0 and 1
        for true_alarm_rate in true_alarm_rates:
            if true_alarm_rate < 0 or true_alarm_rate > 1:
                self.fail("True alarm rates are not between 0 and 1")

class TestFalseAlarmRates(unittest.TestCase):

    def setUp(self):

        # Load simple dsm experiment
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'dsm', 'sthr', 'auroc', 0.02)
        self.experiment = Experiment(experiment_config, 'test')
    
    def test_false_alarm_rates_constant_values(self):
        """Ensures that the false alarm rates don't look wrong"""

        # Get the false alarm rates
        _, false_alarm_rates = self.experiment.true_false_alarm_rates()

        # Check that the false alarm rates are not all 0 or 1
        if (false_alarm_rates == 0).all():
            self.fail("False alarm rates are all 0")
        if (false_alarm_rates == 1).all():
            self.fail("False alarm rates are all 1")

    def test_false_alarm_rates_decreasing(self):
        """Ensures that the false alarm rates are decreasing with higher thresholds"""

        # Get the false alarm rates
        _, false_alarm_rates = self.experiment.true_false_alarm_rates()

        # Check that the false alarm rates are increasing
        for i in range(len(false_alarm_rates) - 1):
            if false_alarm_rates[i] < false_alarm_rates[i + 1]:
                self.fail("False alarm rates are not decreasing")

    def test_false_alarm_rates_boundaries(self):
        """Check that the false alarm rates are all between 0 and 1"""
            
        # Get the false alarm rates
        _, false_alarm_rates = self.experiment.true_false_alarm_rates()

        # Check that the false alarm rates are between 0 and 1
        for false_alarm_rate in false_alarm_rates:
            if false_alarm_rate < 0 or false_alarm_rate > 1:
                self.fail("False alarm rates are not between 0 and 1")

    def test_false_alarm_rates_limits(self):
        """Check that the false alarm rate is 0 when the threshold is 1 and 1 when the threshold is 0"""

        # Get the false alarm rates
        _, false_alarm_rates = self.experiment.true_false_alarm_rates()

        # Check that the false alarm rates are 0 when the threshold is 1 and 1 when the threshold is 0
        if false_alarm_rates[0] != 1:
            self.fail("False alarm rate is not 1 when threshold is 0")
        if false_alarm_rates[-1] != 0:
            self.fail("False alarm rate is not 0 when threshold is 1")


class TestAllCombos(unittest.TestCase):

    # Test for every combination of model, alarm type, and metric
    # TODO: flesh this out
    # model_list = ['cph', 'dcph', 'dsm', 'rf', 'km']
    # alarm_type_list = ['sthr', 'hyst']
    # metric_list = ['auroc', 'auwtc']
    
    model_list = ['rf', 'dsm']
    alarm_type_list = ['sthr']
    metric_list = ['auroc']
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
        
        model = 'rf'
        alarm_type = 'sthr'
        model_metric = 'auroc'
        min_required_warning_time = 0.02
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, model, alarm_type, model_metric, min_required_warning_time)
        experiment = Experiment(experiment_config, 'test')

        experiment.evaluate_metric(model_metric)


class TestCriticalMetric(unittest.TestCase):

    def setUp(self):
        """Set up the test case
        """

        # Load simple RF experiment
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'rf', 'sthr', 'auroc', 0.02)
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

    def test_no_nan(self):

        # Get the metric
        general_false_alarm_rates, general_avg_warning_times, general_std_warning_times = self.experiment.warning_time_vs_false_alarm_rate(horizon=0.05, required_warning_time=0.02)    

        # Check that there are no NaNs
        if (np.isnan(general_false_alarm_rates)).any():
            self.fail("NaNs in false alarm rates")
        if (np.isnan(general_avg_warning_times)).any():
            self.fail("NaNs in average warning times")
        if (np.isnan(general_std_warning_times)).any():
            self.fail("NaNs in standard deviation of warning times")

class TestWarningTimesList(unittest.TestCase):

    def test_no_negative_warning_times(self):
        """Ensure that there are no negative warning times
        """

        # Load hysteresis rf experiment
        required_warning_time = 0.02
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'rf', 'sthr', 'auroc', required_warning_time)
        experiment = Experiment(experiment_config, 'test')

        # Get the warning times list
        warning_times_list = experiment.get_warning_times_list()

        # Check that there are no negative warning times
        for warning_times in warning_times_list:
            for warning_time in warning_times:
                self.assertGreaterEqual(warning_time, 0)

    def test_warning_times_nonzero_with_true_positives(self):
        """In the case when the true positive rate is greater than zero for a given false positve rate,
        ensure that the average warning time is greater than zero"""

        # Load dsm simple threshold experiment
        required_warning_time = 0.1
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'dsm', 'sthr', 'auroc', required_warning_time)
        self.experiment = Experiment(experiment_config, 'test')

        # Get the true alarm rate list
        true_alarm_rates, _ = self.experiment.true_false_alarm_rates()
        
        # Get the warning time list
        warning_times = self.experiment.get_warning_times_list(required_warning_time=required_warning_time)

        # Get the average warning times
        avg_warning_times = np.mean(warning_times, axis=1)

        # Check that the warning times are empty when the true alarm rate is 0
        for i, true_alarm_rate in enumerate(true_alarm_rates):
            if true_alarm_rate != 0:
                if avg_warning_times[i] == 0:
                    self.fail(f"There are no warning times when there exist true positives {i}")

    def test_warning_times_increasing(self):
        """Ensure that the warning times are increasing with increasing false alarm rates
        """

        # Load dsm simple threshold experiment
        required_warning_time = 0.1
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'dsm', 'sthr', 'auroc', required_warning_time)
        self.experiment = Experiment(experiment_config, 'test')

        general_false_alarm_rates, general_avg_warning_times, _ = self.experiment.warning_time_vs_false_alarm_rate()    

        # Check that the warning times are increasing
        for i in range(len(general_false_alarm_rates) - 1):
            if general_avg_warning_times[i] > general_avg_warning_times[i + 1]:
                self.fail("Warning times are not increasing")




