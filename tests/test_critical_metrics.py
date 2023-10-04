import unittest
import numpy as np

from disruption_survival_analysis.critical_metrics import compute_metrics_vs_thresholds

class TestComputeMetricsVsThresholds(unittest.TestCase):

    def test_simple_case(self):
        # Make two shots, one disruptive and one non-disruptive
        
        times = np.array([0, 1, 2, 3, 4, 5])
        
        shot_1_risk = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        shot_2_risk = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.2])

        shot_1_prediction = {'risk': shot_1_risk,
                             'time': times}
        shot_2_prediction = {'risk': shot_2_risk,
                             'time': times}
        
        predictions = [shot_1_prediction, shot_2_prediction]

        shot_1_outcome = {'disruption_time': 5,
                          'disrupted': True}
        shot_2_outcome = {'disruption_time': np.nan,
                          'disrupted': False}
        
        outcomes = [shot_1_outcome, shot_2_outcome]

        required_warning_time = 1.9

        unique_thresholds = np.unique(np.concatenate((shot_1_risk, shot_2_risk)))

        true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times = compute_metrics_vs_thresholds(predictions, outcomes, required_warning_time, unique_thresholds)

        # Check that the length of these arrays lines up with the length of the thresholds
        if len(true_alarm_rates) != len(unique_thresholds):
            self.fail("Length of True Alarm Rates does not match length of Thresholds")
        if len(false_alarm_rates) != len(unique_thresholds):
            self.fail("Length of False Alarm Rates does not match length of Thresholds")
        if len(avg_warning_times) != len(unique_thresholds):
            self.fail("Length of Average Warning Times does not match length of Thresholds")
        if len(std_warning_times) != len(unique_thresholds):
            self.fail("Length of Standard Deviation of Warning Times does not match length of Thresholds")

        # Check that the true alarm rate is always 0 or 1
        if not np.all(np.logical_or(true_alarm_rates == 0, true_alarm_rates == 1)):
            self.fail("True Alarm Rate should always be 0 or 1 in this simple case")
        # Check that the false alarm rate is always 0 or 1
        if not np.all(np.logical_or(false_alarm_rates == 0, false_alarm_rates == 1)):
            self.fail("False Alarm Rate should always be 0 or 1 in this simple case")
        # Check that the average warning time is always greater than or equal to 0
        if not np.all(avg_warning_times >= 0):
            self.fail("Average Warning Time should always be >= 0")
    
