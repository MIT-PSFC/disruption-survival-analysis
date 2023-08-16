import unittest
import pandas as pd

from disruption_survival_analysis.experiment_utils import *
from disruption_survival_analysis.manage_datasets import load_dataset

# Labeling data tests

class TestLabelShotData(unittest.TestCase):
    """Tests for the function label_shot_data"""

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
        labels = label_shot_data(shot_data, False, 0.2)

        # Check that the returned data is the same length as the input data
        self.assertEqual(len(labels), 5)

        # Check that the sum of the labels is 0
        self.assertEqual(labels.sum(), 0)

class TestMakeShotLifetimeCurve(unittest.TestCase):
    """Tests for the function make_shot_lifetime_curve"""

    def setUp(self):
        self.data = load_dataset("synthetic", "test", "train")

    def test_curve_values_disruptive(self):
        """Test that the curve values are correct for disruptive shots"""

        self.fail("Not implemented")

    def test_curve_values_non_disruptive(self):
        """Test that the curve values are correct for non-disruptive shots"""

        self.fail("Not implemented")

    def test_curve_length(self):
        """Test that the curve has the correct length"""

        self.fail("Not implemented")

# Alarm time tests

class TestCalculateAlarmTimes(unittest.TestCase):
    """Tests for the function calculate_alarm_times"""

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

class TestCalculateAlarmTimesHysteresis(unittest.TestCase):
    """Tests for the function calculate_alarm_times_hysteresis"""

class TestCalculateAlarmTimesEttd(unittest.TestCase):
    """Tests for the function calculate_alarm_times_ettd"""

# Evaluation metric tests

class TestTimesliceMicroAverage(unittest.TestCase):
    """Tests for the function timeslice_micro_average"""

class TestAreaUnderCurve(unittest.TestCase):
    """Tests for the function area_under_curve"""

class TestCalculateF1Scores(unittest.TestCase):
    """Tests for the function calculate_f1_scores"""

class TestExpectedTimeToDisruptionIntegral(unittest.TestCase):
    """Tests for the function expected_time_to_disruption_integral"""

# Misc function tests

class TestClumpManyToOneStatistics(unittest.TestCase):
    """Tests for the function clump_many_to_one_statistics"""

    def test_clump_many_to_one_statistics_single_array(self):

        # Create numpy array of the warning times
        warning_times = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        true_alarm_rates = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4]

        unique_true_alarm_rates, avg_warning_times, std_warning_times = clump_many_to_one_statistics(true_alarm_rates, warning_times)

        self.assertEqual(len(unique_true_alarm_rates), 4)
        self.assertEqual(len(avg_warning_times), 4)
        self.assertEqual(len(std_warning_times), 4)

        self.assertEqual(unique_true_alarm_rates[0], 0.1)
        self.assertEqual(unique_true_alarm_rates[1], 0.2)
        self.assertEqual(unique_true_alarm_rates[2], 0.3)
        self.assertEqual(unique_true_alarm_rates[3], 0.4)

        self.assertEqual(avg_warning_times[0], np.mean([0.1, 0.2, 0.3]))
        self.assertEqual(avg_warning_times[1], np.mean([0.4, 0.5, 0.6]))
        self.assertEqual(avg_warning_times[2], np.mean([0.7, 0.8]))
        self.assertEqual(avg_warning_times[3], 0.9)

        self.assertEqual(std_warning_times[0], np.std([0.1, 0.2, 0.3]))
        self.assertEqual(std_warning_times[1], np.std([0.4, 0.5, 0.6]))
        self.assertEqual(std_warning_times[2], np.std([0.7, 0.8]))
        self.assertEqual(std_warning_times[3], 0)

    def test_clump_many_to_one_statistics_double_array(self):
        
        # Create numpy arrays of the warning times
        array_1 = np.array([0.1, 0.2, 0.3])
        array_2 = np.array([0.4, 0.5, 0.6])
        array_3 = np.array([0.7, 0.8, 0.9])

        warning_times_list = [array_1, array_2, array_3]
        # Turn warning times into a list of numpy arrays
        warning_times_list = [np.array(warning_times) for warning_times in warning_times_list]

        true_alarm_rates = [0.1, 0.5, 0.1]

        unique_true_alarm_rates, avg_warning_times, std_warning_times = clump_many_to_one_statistics(true_alarm_rates, warning_times_list)

        self.assertEqual(len(unique_true_alarm_rates), 2)
        self.assertEqual(len(avg_warning_times), 2)
        self.assertEqual(len(std_warning_times), 2)

        self.assertEqual(unique_true_alarm_rates[0], 0.1)
        self.assertEqual(unique_true_alarm_rates[1], 0.5)

        first_warns = [0.1, 0.3, 0.4, 0.6, 0.7, 0.9]
        second_warns = [0.2, 0.5, 0.8]

        self.assertEqual(avg_warning_times[0], np.mean(first_warns))
        self.assertEqual(avg_warning_times[1], np.mean(second_warns))

        self.assertEqual(std_warning_times[0], np.std(first_warns))
        self.assertEqual(std_warning_times[1], np.std(second_warns))