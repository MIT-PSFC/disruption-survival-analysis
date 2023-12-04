import unittest
import numpy as np

from disruption_survival_analysis.critical_metrics import compute_metrics_vs_risk_thresholds, compute_metrics_vs_false_alarm_rates, compute_metrics_vs_false_alarm_rates_distribution

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

        true_alarm_rates, false_alarm_rates, avg_warning_times = compute_metrics_vs_risk_thresholds(predictions, outcomes, required_warning_time, unique_thresholds)

        # Check that the length of these arrays lines up with the length of the thresholds
        if len(true_alarm_rates) != len(unique_thresholds):
            self.fail("Length of True Alarm Rates does not match length of Thresholds")
        if len(false_alarm_rates) != len(unique_thresholds):
            self.fail("Length of False Alarm Rates does not match length of Thresholds")
        if len(avg_warning_times) != len(unique_thresholds):
            self.fail("Length of Average Warning Times does not match length of Thresholds")

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

        true_alarm_rates, false_alarm_rates, avg_warning_times = compute_metrics_vs_risk_thresholds(predictions, outcomes, required_warning_time, unique_thresholds)

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
        #if not np.all(std_warning_times == correct_std_warning_times):
        #    self.fail("Standard Deviation of Warning Time is incorrect")

class TestComputeMetricsVsFalseAlarmRates(unittest.TestCase):

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

        unique_false_alarm_rates, avg_true_alarm_rates, avg_warning_times = compute_metrics_vs_false_alarm_rates(predictions, outcomes, required_warning_time, unique_thresholds, 'sthr')

        correct_unique_false_alarm_rates = np.array([0, 1])
        correct_avg_true_alarm_rates = np.array([0.5, 1])
        correct_avg_warning_times = np.array([1, 2])

        # Check that all these values are close
        if not np.allclose(unique_false_alarm_rates, correct_unique_false_alarm_rates):
            self.fail("Unique False Alarm Rates are not close")
        if not np.allclose(avg_true_alarm_rates, correct_avg_true_alarm_rates):
            self.fail("Average True Alarm Rates are not close")
        if not np.allclose(avg_warning_times, correct_avg_warning_times):
            self.fail("Average Warning Times are not close")


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

        false_alarm_rates, avg_true_alarm_rates, avg_warning_times = compute_metrics_vs_false_alarm_rates(predictions, outcomes, required_warning_time, unique_thresholds, 'sthr')

        # Check that the false alarm rates are correct
        correct_false_alarm_rates = np.unique([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0])
        if not np.all(false_alarm_rates == correct_false_alarm_rates):
            self.fail("False Alarm Rate is incorrect")

        #corrct_true_alarm_rates = np.array([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0, 0, 0])
        tar_groups = [
            np.array([0, 0]),
            np.array([1, 1, 1, 1, 1, .5, .5, 0]),
            np.array([1])
        ]

        correct_true_alarm_rates = np.zeros(len(correct_false_alarm_rates))
        for i in range(len(correct_false_alarm_rates)):
            correct_true_alarm_rates[i] = np.mean(tar_groups[i])

        if not np.isclose(avg_true_alarm_rates, correct_true_alarm_rates).all():
            self.fail("True Alarm Rate is incorrect")

        #shot_0_warning_times = np.array([80, 70, 60, 50, 40, 30, 20, 10, 0, 0, 0])
        #shot_1_warning_times = np.array([70, 70, 70, 70, 50, 50, 50, 50, 0, 0, 0])

        warn_groups = [
            np.array([0, 0, 0, 0]),
            np.array([70, 60, 50, 40, 30, 20, 10, 0, 70, 70, 70, 50, 50, 50, 50, 0]),
            np.array([80, 70])
        ]

        correct_avg_warning_times = np.zeros(len(correct_false_alarm_rates))
        for i in range(len(correct_false_alarm_rates)):
            correct_avg_warning_times[i] = np.mean(warn_groups[i])

        # For each threshold, check that the average warning time is correct
        if not np.isclose(avg_warning_times, correct_avg_warning_times).all():
            self.fail("Average Warning Time is incorrect")

class TestComputeMetricsDistributionAgreement(unittest.TestCase):

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

        unique_false_alarm_rates, avg_true_alarm_rates, avg_warning_times = compute_metrics_vs_false_alarm_rates(predictions, outcomes, required_warning_time, unique_thresholds, 'sthr')

        unique_false_alarm_rates_distribution, tar_metrics, warn_metrics = compute_metrics_vs_false_alarm_rates_distribution(predictions, outcomes, required_warning_time, unique_thresholds, 'sthr')

        # Ensure that all calculations are close
        if not np.allclose(unique_false_alarm_rates, unique_false_alarm_rates_distribution):
            self.fail("Unique False Alarm Rates are not close")
        if not np.allclose(avg_true_alarm_rates, tar_metrics['avg']):
            self.fail("Average True Alarm Rates are not close")
        if not np.allclose(avg_warning_times, warn_metrics['avg']):
            self.fail("Average Warning Times are not close")

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

        unique_false_alarm_rates, avg_true_alarm_rates, avg_warning_times = compute_metrics_vs_false_alarm_rates(predictions, outcomes, required_warning_time, unique_thresholds, 'sthr')

        unique_false_alarm_rates_distribution, tar_metrics, warn_metrics = compute_metrics_vs_false_alarm_rates_distribution(predictions, outcomes, required_warning_time, unique_thresholds, 'sthr')

        # Ensure that all calculations are close
        if not np.allclose(unique_false_alarm_rates, unique_false_alarm_rates_distribution):
            self.fail("Unique False Alarm Rates are not close")
        if not np.allclose(avg_true_alarm_rates, tar_metrics['avg']):
            self.fail("Average True Alarm Rates are not close")
        if not np.allclose(avg_warning_times, warn_metrics['avg']):
            self.fail("Average Warning Times are not close")

    def test_calculation_exact_all(self):
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

        false_alarm_rates, true_alarm_metrics, warning_time_metrics = compute_metrics_vs_false_alarm_rates_distribution(predictions, outcomes, required_warning_time, unique_thresholds, 'sthr')

        # Check that the false alarm rates are correct
        correct_false_alarm_rates = np.unique([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0])
        if not np.all(false_alarm_rates == correct_false_alarm_rates):
            self.fail("False Alarm Rate is incorrect")

        #corrct_true_alarm_rates = np.array([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0, 0, 0])
        tar_groups = [
            np.array([0, 0]),
            np.array([1, 1, 1, 1, 1, .5, .5, 0]),
            np.array([1])
        ]

        avg_true_alarm_rates = np.zeros(len(correct_false_alarm_rates))
        std_true_alarm_rates = np.zeros(len(correct_false_alarm_rates))
        med_true_alarm_rates = np.zeros(len(correct_false_alarm_rates))
        iq1_true_alarm_rates = np.zeros(len(correct_false_alarm_rates))
        iq3_true_alarm_rates = np.zeros(len(correct_false_alarm_rates))
        for i in range(len(correct_false_alarm_rates)):
            avg_true_alarm_rates[i] = np.mean(tar_groups[i])
            std_true_alarm_rates[i] = np.std(tar_groups[i])
            med_true_alarm_rates[i] = np.median(tar_groups[i])
            iq1_true_alarm_rates[i] = np.quantile(tar_groups[i], 0.25)
            iq3_true_alarm_rates[i] = np.quantile(tar_groups[i], 0.75)

        if not np.isclose(true_alarm_metrics['avg'], avg_true_alarm_rates).all():
            self.fail("Average True Alarm Rate is incorrect")
        if not np.isclose(true_alarm_metrics['std'], std_true_alarm_rates).all():
            self.fail("Standard Deviation of True Alarm Rate is incorrect")
        if not np.isclose(true_alarm_metrics['med'], med_true_alarm_rates).all():
            self.fail("Median of True Alarm Rate is incorrect")
        if not np.isclose(true_alarm_metrics['iq1'], iq1_true_alarm_rates).all():
            self.fail("First Quartile of True Alarm Rate is incorrect")
        if not np.isclose(true_alarm_metrics['iq3'], iq3_true_alarm_rates).all():
            self.fail("Third Quartile of True Alarm Rate is incorrect")

        #shot_0_warning_times = np.array([80, 70, 60, 50, 40, 30, 20, 10, 0, 0, 0])
        #shot_1_warning_times = np.array([70, 70, 70, 70, 50, 50, 50, 50, 0, 0, 0])

        warn_groups = [
            np.array([0, 0, 0, 0]),
            np.array([70, 60, 50, 40, 30, 20, 10, 0, 70, 70, 70, 50, 50, 50, 50, 0]),
            np.array([80, 70])
        ]

        avg_warning_times = np.zeros(len(correct_false_alarm_rates))
        std_warning_times = np.zeros(len(correct_false_alarm_rates))
        med_warning_times = np.zeros(len(correct_false_alarm_rates))
        iq1_warning_times = np.zeros(len(correct_false_alarm_rates))
        iq3_warning_times = np.zeros(len(correct_false_alarm_rates))
        for i in range(len(correct_false_alarm_rates)):
            avg_warning_times[i] = np.mean(warn_groups[i])
            std_warning_times[i] = np.std(warn_groups[i])
            med_warning_times[i] = np.median(warn_groups[i])
            iq1_warning_times[i] = np.quantile(warn_groups[i], 0.25)
            iq3_warning_times[i] = np.quantile(warn_groups[i], 0.75)

        # Doing iqm by hand
        iqm_warning_times = np.array([
            np.mean([0, 0]),
            np.mean([60, 50, 50, 50, 50, 50, 40, 30]),
            np.mean([80, 70])
        ])

        # For each false alarm rate, check that the average warning time is correct
        if not np.isclose(warning_time_metrics['avg'], avg_warning_times).all():
            self.fail("Average Warning Time is incorrect")
        if not np.isclose(warning_time_metrics['std'], std_warning_times).all():
            self.fail("Standard Deviation of Warning Time is incorrect")
        if not np.isclose(warning_time_metrics['med'], med_warning_times).all():
            self.fail("Median of Warning Time is incorrect")
        if not np.isclose(warning_time_metrics['iq1'], iq1_warning_times).all():
            self.fail("First Quartile of Warning Time is incorrect")
        if not np.isclose(warning_time_metrics['iq3'], iq3_warning_times).all():
            self.fail("Third Quartile of Warning Time is incorrect")
        if not np.isclose(warning_time_metrics['iqm'], iqm_warning_times).all():
            self.fail("Interquartile Mean of Warning Time is incorrect")