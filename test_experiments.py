"""Test functions to ensure that the data preprocessing works as expected
"""
import unittest

from run_models import load_model
from preprocess_datasets import get_disruptive_shot_list
from evaluate_performance import *

from DisruptionPredictors import DisruptionPredictorSM

class TestTimeToDetection(unittest.TestCase):

    def setUp(self):
        """Set up the test case
        """
        
        # Specify testing parameters
        self.device = 'synthetic'
        self.dataset = 'synthetic100'
        self.numeric_feats = ['ip', 'Wmhd', 'n_e', 'kappa', 'li']
        self.horizon = 0.2
        self.thresholds = np.linspace(0, 1, 100)

        # Load model
        self.model, self.transformer = load_model('cph', self.device, self.dataset)
        self.predictor = DisruptionPredictorSM("Cox Proportional Hazards", self.model, self.numeric_feats, self.transformer)

        # Load data
        self.data = load_benchmark_data(self.predictor, self.device, self.dataset+'_test')

    
    def test_calc_num_disrupt(self):
        """Test that the number of disruptions is calculated correctly
        """
        # Get the number of disruptions in the test set
        true_count = len(get_disruptive_shot_list(self.device, self.dataset+'_test'))

        # Calculate the number of disruptions in the test set
        calc_num_disrupt_result = calc_num_disrupt(self.data)

        self.assertEqual(true_count, calc_num_disrupt_result)

    def test_detection_times(self):
        """Ensure distance between detection time and disruption time is always decreasing or None"""

        _, _, detection_times_array = calc_tp_fp_times(self.predictor, self.horizon, self.data, self.thresholds)

        # Check that the detection times are always decreasing until they are None
        for detection_times in detection_times_array:
            for i in range(len(detection_times)-1):
                if detection_times[i] is None:
                    self.assertTrue(all([x is None for x in detection_times[i:]]))
                    break
                else:
                    self.assertLessEqual(detection_times[i+1], detection_times[i])


    def test_false_positive_rates(self):
        """Test that the false positive rates are always decreasing
        As threshold increases, false positive rate should decrease
        """
        
        # Get false positives
        _, false_positives, _ = calc_tp_fp_times(self.predictor, self.horizon, self.data, self.thresholds)

        # Get number of disruptions
        num_disrupt = calc_num_disrupt(self.data)

        # Check that the false positive rates are always decreasing
        false_positive_rates = np.sum(false_positives, axis=0)/num_disrupt

        for i in range(len(false_positive_rates)-1):
            self.assertLessEqual(false_positive_rates[i+1], false_positive_rates[i])

    def test_true_detection_ordering(self):
        """Ensure that the mean detection times are always decreasing
        As threshold increases, there should be less time between detection and disruption
        """

        _, mean_detection_times, _ = benchmark_true_detection(self.predictor, self.horizon, self.device, self.dataset+'_test')

        # Check that the mean detection times are always decreasing
        for i in range(len(mean_detection_times)-1):
            self.assertLessEqual(mean_detection_times[i+1], mean_detection_times[i])


    def test_true_detection_sizes(self):
        """Ensure all arrays returned by benchmark_true_detection are the same size"""

        false_positive_rates, mean_detection_times, std_detection_times = benchmark_true_detection(self.predictor, self.horizon, self.device, self.dataset+'_test')

        self.assertEqual(len(false_positive_rates), len(mean_detection_times))
        self.assertEqual(len(false_positive_rates), len(std_detection_times))