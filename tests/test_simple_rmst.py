import unittest

import os
import dill

from tests.test_manage_datasets import TEST_DEVICE, TEST_DATASET_PATH, TEST_WARNING_TIME

from simple_rmst_integral import main

class TestSimpleRMST_RF_STHR_AUROC(unittest.TestCase):

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'rf'
        self.alarm_type = 'sthr'
        self.metric = 'auroc'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        self.rmst_path = f"results/{self.device}/{self.dataset_path}/simple_rmst"
        self.category_path = f"{self.rmst_path}/{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms"
        self.saved_file_path = f"{self.category_path}/all_rmst_results.pkl"

    def test_0_rmst_integral_completes(self):
        """Ensure that the rmst integral job completes"""

        # Remove bootstrap file if it exists
        try:
            os.remove(self.saved_file_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms)

        # Check that the bootstrap file exists
        self.assertTrue(os.path.exists(self.saved_file_path))

    def test_1_values_exist(self):
        """Ensure the saved values are not empty"""

        # Load pickle file
        with open(self.saved_file_path, 'rb') as f:
            results = dill.load(f)

        disruptive_rmst_diffs = results['disruptive_rmst_diffs']
        non_disruptive_rmst_diffs = results['non_disruptive_rmst_diffs']

        # Check that the values are not empty
        self.assertTrue(len(disruptive_rmst_diffs) > 0)
        self.assertTrue(len(non_disruptive_rmst_diffs) > 0)

        for result_group in ['disruptive_results', 'non_disruptive_results', 'all_results']:
            for metric in ['avg', 'std', 'med', 'iq1', 'iq3', 'iqm']:
                self.assertTrue(results[result_group][metric] is not None)

class TestSimpleRMST_CMOD_WONK(unittest.TestCase):

    def setUp(self):
        self.device = 'cmod'
        self.dataset_path = 'paper_4/stack_10'
        self.model_type = 'dcph'
        self.alarm_type = 'sthr'
        self.metric = 'auroc'
        self.required_warning_time_ms = 50

        self.rmst_path = f"results/{self.device}/{self.dataset_path}/simple_rmst"
        self.category_path = f"{self.rmst_path}/{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms"
        self.saved_file_path = f"{self.category_path}/all_rmst_results.pkl"

    def test_0_rmst_integral_completes(self):
        """Ensure that the rmst integral job completes"""

        # Remove bootstrap file if it exists
        try:
            os.remove(self.saved_file_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms)

        # Check that the bootstrap file exists
        self.assertTrue(os.path.exists(self.saved_file_path))

    def test_1_values_exist(self):
        """Ensure the saved values are not empty"""

        # Load pickle file
        with open(self.saved_file_path, 'rb') as f:
            results = dill.load(f)

        disruptive_rmst_diffs = results['disruptive_rmst_diffs']
        non_disruptive_rmst_diffs = results['non_disruptive_rmst_diffs']

        # Check that the values are not empty
        self.assertTrue(len(disruptive_rmst_diffs) > 0)
        self.assertTrue(len(non_disruptive_rmst_diffs) > 0)

        for result_group in ['disruptive_results', 'non_disruptive_results', 'all_results']:
            for metric in ['avg', 'std', 'med', 'iq1', 'iq3', 'iqm']:
                self.assertTrue(results[result_group][metric] is not None)

        # Ensure that the RMST values are 'good' (not NaN or Inf or really big)
        for result_group in ['disruptive_results', 'non_disruptive_results', 'all_results']:
            for metric in ['avg', 'std', 'med', 'iq1', 'iq3', 'iqm']:
                self.assertTrue(results[result_group][metric] < 1)
