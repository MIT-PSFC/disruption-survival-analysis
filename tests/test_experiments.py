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

class TestCriticalMetric(unittest.TestCase):

    def setUp(self):
        """Set up the test case
        """

        # Load simple DSM experiment
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'dsm', 'sthr', 'auroc', 0.02)
        self.experiment = Experiment(experiment_config, 'test')

    def test_compute_metric(self):
        """Test that the metric is calculated correctly
        """

        # Get the metric both ways
        direct_false_alarm_rates, direct_avg_warning_times, direct_std_warning_times = self.experiment.compute_critical_metric(horizon=0.05, required_warning_time=0.02)
        general_false_alarm_rates, general_avg_warning_times, general_std_warning_times = self.experiment.warning_time_vs_false_alarm_rate(horizon=0.05, required_warning_time=0.02)

        # Check that the metric is calculated correctly
        self.assertEqual(direct_false_alarm_rates, general_false_alarm_rates)
        self.assertEqual(direct_avg_warning_times, general_avg_warning_times)
        self.assertEqual(direct_std_warning_times, general_std_warning_times)

# class TestExperimentsAlarms(unittest.TestCase):

#     def setUp(self):
#         """Set up the test case
#         """

#         # Load simple DSM experiment
#         self.experiment = Experiment()

#     def test_alarm_times_shapes(self):
#         """Ensure that the alarms and times arrays have the correct shapes"""

#         # Get the alarm times
#         alarm_times = self.experiment.get_alarm_times(self.horizon)

#         # Check that the arrays have the correct shapes
#         # Should be the same length as the number of shots
#         # and the second dimension is the number of thresholds
#         self.assertEqual(alarm_times.shape, (self.experiment.get_num_shots(), len(self.experiment.thresholds)))
        
#     def test_true_false_alarms_shapes(self):

#         # Get the true and false alarms
#         true_alarms, false_alarms = self.experiment.get_true_false_alarms(self.horizon, self.required_warning_time)

#         # Check that the arrays have the correct shapes
#         # Should be the same length as the number of shots
#         # and the second dimension is the number of thresholds
#         self.assertEqual(true_alarms.shape, (self.experiment.get_num_shots(), len(self.experiment.thresholds)))
#         self.assertEqual(false_alarms.shape, (self.experiment.get_num_shots(), len(self.experiment.thresholds)))


# class TestWarningTimeMethods(unittest.TestCase):

#     def test_warning_times_list_length(self):
#         """Ensure that the warning times array has the correct length"""

#         # Get the warning times list
#         warning_times_list = self.experiment.get_warning_times_list(self.horizon)

#         # Check that the warning times list has the correct length
#         # Should be the same length as the number of disruptive shots
#         # as the second dimension is the number of thresholds
#         self.assertEqual(len(warning_times_list), self.experiment.get_num_disruptive_shots())

