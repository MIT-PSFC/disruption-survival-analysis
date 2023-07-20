"""Test functions to ensure that the data preprocessing works as expected
"""
import unittest
import numpy as np
import pandas as pd

from Experiments import Experiment
from DisruptionPredictors import DisruptionPredictorSM

from run_models import load_model
from experiment_utils import label_shot_data, calculate_alarm_times
from manage_datasets import load_dataset, load_disruptive_shot_list, load_non_disruptive_shot_list

TEST_DEVICE = 'cmod'
TEST_DATASET_PATH = 'random_256_shots_60%_flattop'

class TestSimpleFunctions(unittest.TestCase):

    def setUp(self):
        """Set up the test case
        """
        
        # Specify testing parameters
        self.horizon = 0.2

        # Load simple CPH model
        self.model, self.transformer, self.numeric_feats = load_model('cph', TEST_DEVICE, TEST_DATASET_PATH)
        self.predictor = DisruptionPredictorSM("Cox Proportional Hazards", self.model, self.numeric_feats, self.transformer)
        self.experiment = Experiment(TEST_DEVICE, TEST_DATASET_PATH, self.predictor, name='CPH')

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
            if shot == disruptive_shots[next_disruptive_shot]:
                next_disruptive_shot += 1
            elif shot == non_disruptive_shots[next_non_disruptive_shot]:
                next_non_disruptive_shot += 1
            else:
                self.fail("Shot lists are not ordered consistently")


class TestExperimentUtils(unittest.TestCase):

    def test_label_shot_data_disruptive(self):
        """Ensure that the label_shot_data function labels disruptive shots correctly
        """

        # Create a Pandas dataframe of shot data with a single shot and only the 'time' column
        shot_data = pd.DataFrame({'time': [0.1, 0.2, 0.3, 0.4, 0.5]})

        # Call the label_shot_data function as a disruptive shot
        labels = label_shot_data(shot_data, True, 0.2)

        # Check that the returned data is the same length as the input data
        self.assertEqual(len(labels), 5)

        # Check that the first three labels are 0 and the last two are 1
        self.assertEqual(labels[0:3].sum(), 0)
        self.assertEqual(labels[3:].sum(), 2)

    def test_label_shot_data_non_disruptive(self):
        """Ensure that the label_shot_data function labels non-disruptive shots correctly
        """

        # Create a Pandas dataframe of shot data with a single shot and only the 'time' column
        shot_data = pd.DataFrame({'time': [0.1, 0.2, 0.3, 0.4, 0.5]})

        # Call the label_shot_data function as a non-disruptive shot
        labels = label_shot_data(shot_data, False, 0)

        # Check that the returned data is the same length as the input data
        self.assertEqual(len(labels), 5)

        # Check that the sum of the labels is 0
        self.assertEqual(labels.sum(), 0)

    def test_calculate_alarm_times_exact(self):
        """Ensure that the calculate_alarm_times function returns the correct times
        This function tests the case where the risk exceeds the threshold exactly
        """

        # Create a Pandas dataframe of risks at different times
        risk_at_time = pd.DataFrame({'time': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'risk': [0.01, 0.11, 0.71, 0.21, 0.81, 0.41]})

        # Calculate the alarm times
        alarm_times = calculate_alarm_times(risk_at_time, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        # Check that the alarm times are correct
        self.assertEqual(alarm_times[0], 0)
        self.assertEqual(alarm_times[1], 0.1)
        self.assertEqual(alarm_times[2], 0.2)
        self.assertEqual(alarm_times[3], 0.2)
        self.assertEqual(alarm_times[4], 0.2)
        self.assertEqual(alarm_times[5], 0.2)
        self.assertEqual(alarm_times[6], 0.2)
        self.assertEqual(alarm_times[7], 0.2)
        self.assertEqual(alarm_times[8], 0.4)
        self.assertEqual(alarm_times[9], None)
        self.assertEqual(alarm_times[10], None)


class TestExperimentsAlarms(unittest.TestCase):

    def setUp(self):
        """Set up the test case
        """
        
        # Specify testing parameters
        self.horizon = 0.2

        # Load simple CPH model
        self.model, self.transformer, self.numeric_feats = load_model('cph', TEST_DEVICE, TEST_DATASET_PATH)
        self.predictor = DisruptionPredictorSM("Cox Proportional Hazards", self.model, self.numeric_feats, self.transformer)
        self.experiment = Experiment(TEST_DEVICE, TEST_DATASET_PATH, self.predictor, name='CPH')

    def test_alarms_times_shapes(self):
        """Ensure that the alarms and times arrays have the correct shapes"""

        # Get the alarms and  times
        true_alarms, false_alarms, alarm_times = self.experiment.get_alarms_times(self.horizon)

        # Check that the arrays have the correct shapes
        # Should be the same length as the number of shots
        # and the second dimension is the number of thresholds
        self.assertEqual(true_alarms.shape, (self.experiment.get_num_shots(), len(self.experiment.thresholds)))
        self.assertEqual(false_alarms.shape, (self.experiment.get_num_shots(), len(self.experiment.thresholds)))
        self.assertEqual(alarm_times.shape, (self.experiment.get_num_shots(), len(self.experiment.thresholds)))
        

    def test_warning_times_list_length(self):
        """Ensure that the warning times array has the correct length"""

        # Get the warning times list
        warning_times_list = self.experiment.get_warning_times_list(self.horizon)

        # Check that the warning times list has the correct length
        # Should be the same length as the number of disruptive shots
        # as the second dimension is the number of thresholds
        self.assertEqual(len(warning_times_list), self.experiment.get_num_disruptive_shots())

    def test_warning_times_ordering(self):
        """Ensure that the warning times for a given shot are always decreasing
        As threshold increases, there should be less time between detection and disruption
        """

        warning_times_list = self.experiment.get_warning_times_list(self.horizon)

        # Check that the warning times are always decreasing until there aren't any more
        for warning_times in warning_times_list:
            for i in range(len(warning_times)-1):
                self.assertLessEqual(warning_times[i+1], warning_times[i])

    def test_mean_warning_time_vs_threshold_ordering(self):
        """Ensure that the mean warning times are always decreasing
        As threshold increases, there should be less time between detection and disruption
        """

        _, mean_warning_times, _ = benchmark_warning_time(self.predictor, self.horizon, self.device, self.dataset+'_test')

        # Check that the mean detection times are always decreasing
        for i in range(len(mean_warning_times)-1):
            self.assertLessEqual(mean_warning_times[i+1], mean_warning_times[i])


    def test_warning_time_sizes(self):
        """Ensure all arrays returned by benchmark_warning_time are the same size"""

        false_positive_rates, mean_warning_times, std_warning_times = benchmark_warning_time(self.predictor, self.horizon, self.device, self.dataset+'_test')

        self.assertEqual(len(false_positive_rates), len(mean_warning_times))
        self.assertEqual(len(false_positive_rates), len(std_warning_times))


    def test_calculate_disruption_times_increasing(self):
        """Ensure that the disruption times are always increasing"""
        
        # Pick some shot number
        shot = self.experiment.get_shot_list()[0]

        # Calculate the risk at time using some horizon
        risk_at_time = self.experiment.get_risk(shot, self.horizon)

        # Calculate the disruption times
        disruption_times = calculate_disruption_times(risk_at_time, np.linspace(0, 1, 100))

        # Check that the disruption times are always increasing until they are none, then all none
        for i in range(len(disruption_times)-1):
            if disruption_times[i] is None:
                self.assertTrue(all([x is None for x in disruption_times[i:]]))
                break
            else:
                self.assertLessEqual(disruption_times[i], disruption_times[i+1])