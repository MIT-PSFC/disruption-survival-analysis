import unittest
import numpy as np

from disruption_survival_analysis.critical_metrics import compute_metrics_vs_risk_thresholds

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

        true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times = compute_metrics_vs_risk_thresholds(predictions, outcomes, required_warning_time, unique_thresholds)

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


    def test_calculation_exact(self):
        # Make a few disruptive and non-disruptive shots, then make sure what is returned is exactly correct

        times =       np.array([0,   10,  20,  30,  40,  50,  60,  70,  80,  90])
        
        shot_0_risk = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        shot_1_risk = np.array([0.4, 0.4, 0.8, 0.2, 0.2, 0.2, 0.8, 0.6])
        shot_2_risk = np.array([0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        shot_3_risk = np.array([0.1, 0.1, 0.3, 0.3, 0.1, 0.1, 0.9, 0.1])

        shot_0_prediction = {'risk': shot_0_risk,
                             'time': times}
        shot_1_prediction = {'risk': shot_1_risk,
                             'time': times[:-2]}
        shot_2_prediction = {'risk': shot_2_risk,
                            'time': times[:-3]}
        shot_3_prediction = {'risk': shot_3_risk,
                            'time': times[:-2]}
        
        
        predictions = [shot_0_prediction, shot_1_prediction, shot_2_prediction, shot_3_prediction]

        shot_0_outcome = {'disruption_time': 90,
                          'disrupted': True}
        shot_1_outcome = {'disruption_time': 70,
                          'disrupted': True}
        shot_2_outcome = {'disruption_time': np.nan,
                          'disrupted': False}
        shot_3_outcome = {'disruption_time': np.nan,
                          'disrupted': False}
        
        outcomes = [shot_0_outcome, shot_1_outcome, shot_2_outcome, shot_3_outcome]

        required_warning_time = 20

        unique_thresholds = np.unique(np.concatenate((shot_0_risk, shot_1_risk, shot_2_risk, shot_3_risk, np.array([0, 1]))))

        true_alarm_rates, false_alarm_rates, avg_warning_times, std_warning_times = compute_metrics_vs_risk_thresholds(predictions, outcomes, required_warning_time, unique_thresholds)

        # For each threshold, check that the true alarm rate is correct
        corrct_true_alarm_rates = np.array([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0, 0, 0])
        if not np.all(true_alarm_rates == corrct_true_alarm_rates):
            self.fail("True Alarm Rate is incorrect")

        # For each threshold, check that the false alarm rate is correct
        correct_false_alarm_rates = np.array([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0])
        if not np.all(false_alarm_rates == correct_false_alarm_rates):
            self.fail("False Alarm Rate is incorrect")

        shot_0_warning_times = np.array([80, 70, 60, 50, 40, 30, 20, 10, 0, 0, 0])
        shot_1_warning_times = np.array([70, 70, 70, 70, 50, 50, 50, 50, 0, 0, 0])

        correct_avg_warning_times = np.zeros(len(unique_thresholds))
        correct_std_warning_times = np.zeros(len(unique_thresholds))
        for i in range(len(unique_thresholds)):
            correct_avg_warning_times[i] = np.mean([shot_0_warning_times[i], shot_1_warning_times[i]])
            correct_std_warning_times[i] = np.std([shot_0_warning_times[i], shot_1_warning_times[i]])

        # For each threshold, check that the average warning time is correct
        if not np.all(avg_warning_times == correct_avg_warning_times):
            self.fail("Average Warning Time is incorrect")

        # For each threshold, check that the standard deviation of the warning time is correct
        if not np.all(std_warning_times == correct_std_warning_times):
            self.fail("Standard Deviation of Warning Time is incorrect")