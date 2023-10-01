import unittest
import numpy as np

from disruption_survival_analysis.critical_metrics import compute_metrics_vs_threshold

TEST_THRESHOLDS = np.linspace(0, 1, 100)

class TestComputeCriticalMetric(unittest.TestCase):

    def test_simple_case(self):

        times = [0, 1, 2, 3, 4, 5]
        # Make two shots, one disruptive and one non-disruptive
        shot_1_risk = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        shot_2_risk = [0.0, 0.0, 0.0, 0.0, 0.2, 0.2]

        shot_1_prediction = {'risk': shot_1_risk,
                             'time': times}
        shot_2_prediction = {'risk': shot_2_risk,
                             'time': times}
        
        predictions = [shot_1_prediction, shot_2_prediction]

        shot_1_outcome = {'disruption_time': 5,
                          'disrupted': True}
        shot_2_outcome = {'disruption_time': np.nan,
                          'disrupted': False}
        
        true_outcomes = [shot_1_outcome, shot_2_outcome]

        required_warning_time = 1.9

        _, false_alarm_rates, avg_warning_times, std_warning_times = compute_metrics_vs_(predictions, true_outcomes, required_warning_time, TEST_THRESHOLDS)

        # Check that the false alarm rates are correct
        # There should only be two false alarm rates, either 0 or 1
        if len(false_alarm_rates) != 2:
            raise ValueError('There should only be two false alarm rates')
        if false_alarm_rates[0] != 0.0:
            raise ValueError('The first false alarm rate should be 0.0')
        if false_alarm_rates[1] != 1:
            raise ValueError('The second false alarm rate should be 1')
        
        # Check that the average warning times are correct
        # There should be two average warning times, since there are two false alarm rates
        # Further, both average warning times should be 2.0
        if len(avg_warning_times) != 2:
            raise ValueError('There should only be two average warning times')
        if avg_warning_times[0] != 1.975:
            raise ValueError('The first average warning time should be 1.975')
        if avg_warning_times[1] != 2.0:
            raise ValueError('The second average warning time should be 2.0')
        
        # Check that the standard deviation of the warning times are correct
        # There should be two standard deviations of the warning times, since there are two false alarm rates
        # Further, both standard deviations of the warning times should be 0.0
        if len(std_warning_times) != 2:
            raise ValueError('There should only be two standard deviations of the warning times')
        if not np.allclose(std_warning_times[0], 0.222205):
            raise ValueError('The first standard deviation of the warning times should be 0.0')
        if std_warning_times[1] != 0.0:
            raise ValueError('The second standard deviation of the warning times should be 0.0')

    def test_three_shot_case(self):
        # Create sample predictions and true outcomes
        predictions = [
            {'risk': np.array([0.1, 0.2, 0.3]), 'time': np.array([1, 2, 3])},
            {'risk': np.array([0.4, 0.5, 0.6]), 'time': np.array([4, 5, 6])},
            {'risk': np.array([0.7, 0.8, 0.9]), 'time': np.array([7, 8, 9])}
        ]
        true_outcomes = [
            {'disruption_time': 3, 'disrupted': True},
            {'disruption_time': np.nan, 'disrupted': False},
            {'disruption_time': 9, 'disrupted': True}
        ]
        required_warning_time = 0.9

        # Call the function under test
        false_alarm_rates, avg_warning_times, std_warning_times = compute_critical_metric(predictions, true_outcomes, required_warning_time, TEST_THRESHOLDS)

        # Check the output
        assert np.allclose(false_alarm_rates, np.array([0, 1]))
        assert np.allclose(avg_warning_times, np.array([0.42857, 1.36363]))
        assert np.allclose(std_warning_times, np.array([0.72843, 0.88140]))

