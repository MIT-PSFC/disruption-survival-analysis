import unittest

import os

from tests.test_manage_datasets import TEST_DEVICE, TEST_DATASET_PATH, TEST_WARNING_TIME

from bootstrap_job import main

BOOTSTRAP_CPUS = 4

class TestBootstrap_RF_STHR_AUROC(unittest.TestCase):

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'rf'
        self.alarm_type = 'sthr'
        self.metric = 'auroc'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        bootstrap_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_bootstrap"
        self.bootstrap_path = f"results/{self.device}/{self.dataset_path}/bootstraps/{bootstrap_name}.pkl"

    def test_a_bootstrap_completes(self):
        """Ensure that the bootstrap job completes"""
        
        # Remove bootstrap file if it exists
        try:
            os.remove(self.bootstrap_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms, BOOTSTRAP_CPUS)

        # Check that the bootstrap file exists
        self.assertTrue(os.path.exists(self.bootstrap_path))

class TestBootstrap_DCPH_STHR_AUROC(unittest.TestCase):

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'dcph'
        self.alarm_type = 'sthr'
        self.metric = 'auroc'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        bootstrap_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_bootstrap"
        self.bootstrap_path = f"results/{self.device}/{self.dataset_path}/bootstraps/{bootstrap_name}.pkl"

    def test_a_bootstrap_completes(self):
        """Ensure that the bootstrap job completes"""
        
        # Remove bootstrap file if it exists
        try:
            os.remove(self.bootstrap_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms)

        # Check that the bootstrap file exists
        self.assertTrue(os.path.exists(self.bootstrap_path))