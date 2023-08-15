"""Test functions to ensure that the data preprocessing works as expected
"""
import unittest
import numpy as np
import pandas as pd


from disruption_survival_analysis.manage_datasets import load_dataset, load_disruptive_shot_list, load_non_disruptive_shot_list
#from disruption_survival_analysis.Experiments import Experiment, make_experiment
from disruption_survival_analysis.DisruptionPredictors import DisruptionPredictorSM

#from disruption_survival_analysis.model_utils import load_model
from disruption_survival_analysis.experiment_utils import label_shot_data, calculate_alarm_times, clump_many_to_one_statistics

TEST_DEVICE = 'synthetic'
TEST_DATASET_PATH = 'synthetic100'

class TestSimpleFunctions(unittest.TestCase):

    def setUp(self):
        """Set up the test case
        """

        self.experiment = make_experiment('config', 'test')
        
        # aaaah

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

class TestExperimentsAlarms(unittest.TestCase):

    def setUp(self):
        """Set up the test case
        """
        
        # Specify testing parameters
        self.horizon = 0.05
        self.required_warning_time = 0.1

        # Load simple CPH model
        self.model, self.numeric_feats = load_model('cph', TEST_DEVICE, TEST_DATASET_PATH)
        self.predictor = DisruptionPredictorSM("Cox Proportional Hazards", self.model, self.numeric_feats)
        self.experiment = Experiment(TEST_DEVICE, TEST_DATASET_PATH, self.predictor, name='CPH')

    def test_alarm_times_shapes(self):
        """Ensure that the alarms and times arrays have the correct shapes"""

        # Get the alarm times
        alarm_times = self.experiment.get_alarm_times(self.horizon)

        # Check that the arrays have the correct shapes
        # Should be the same length as the number of shots
        # and the second dimension is the number of thresholds
        self.assertEqual(alarm_times.shape, (self.experiment.get_num_shots(), len(self.experiment.thresholds)))
        
    def test_true_false_alarms_shapes(self):

        # Get the true and false alarms
        true_alarms, false_alarms = self.experiment.get_true_false_alarms(self.horizon, self.required_warning_time)

        # Check that the arrays have the correct shapes
        # Should be the same length as the number of shots
        # and the second dimension is the number of thresholds
        self.assertEqual(true_alarms.shape, (self.experiment.get_num_shots(), len(self.experiment.thresholds)))
        self.assertEqual(false_alarms.shape, (self.experiment.get_num_shots(), len(self.experiment.thresholds)))

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

    def test_avg_warning_time_vs_threshold_ordering(self):
        """Ensure that the avg warning times are always decreasing
        As threshold increases, there should be less time between detection and disruption
        """

        # Get the avg warning times
        _, avg_warning_times, _ = self.experiment.warning_time_vs_threshold(self.horizon)

        # Check that the avg detection times are always decreasing
        for i in range(len(avg_warning_times)-1):
            self.assertLessEqual(avg_warning_times[i+1], avg_warning_times[i])

    def test_alarm_times_increasing(self):
        """Ensure that the alarm times are always increasing"""
        
        # Get the alarm times for that shot
        alarm_times = self.experiment.get_alarm_times(self.horizon)

        # Check that the alarm times are always increasing until they are Nan, then all Nan
        for shot_index in range(len(alarm_times)):
            for i in range(len(alarm_times[shot_index])-1):
                if np.isnan(alarm_times[shot_index][i]):
                    self.assertTrue(np.isnan(alarm_times[shot_index][i+1]))
                else:
                    if not np.isnan(alarm_times[shot_index][i+1]):
                        self.assertLessEqual(alarm_times[shot_index][i], alarm_times[shot_index][i+1])