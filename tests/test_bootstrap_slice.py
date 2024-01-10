import unittest

import os

from tests.test_manage_datasets import TEST_DEVICE, TEST_DATASET_PATH, TEST_WARNING_TIME

from bootstrap_slice import main

class TestBootstrap_RF_STHR_AUROC(unittest.TestCase):

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'rf'
        self.alarm_type = 'sthr'
        self.metric = 'auroc'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        self.bootstrap_path = f"results/{self.device}/{self.dataset_path}/bootstraps"
        self.category_path = f"{self.bootstrap_path}/{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms"

    def test_0_bootstrap_slice_completes(self):
        """Ensure that the bootstrap slice job completes"""
        
        slice_path = f"{self.category_path}/slice_0.pkl"

        # Remove bootstrap file if it exists
        try:
            os.remove(slice_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms, 0)

        # Check that the bootstrap file exists
        self.assertTrue(os.path.exists(self.bootstrap_path))

    def test_1_bootstrap_slice_completes(self):
        """Ensure that the bootstrap slice job completes"""
        
        slice_path = f"{self.category_path}/slice_1.pkl"

        # Remove bootstrap file if it exists
        try:
            os.remove(slice_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms, 1)

        # Check that the bootstrap file exists
        self.assertTrue(os.path.exists(self.bootstrap_path))

    def test_2_bootstrap_slice_completes(self):
        """Ensure that the bootstrap slice job completes"""
        
        slice_path = f"{self.category_path}/slice_2.pkl"

        # Remove bootstrap file if it exists
        try:
            os.remove(slice_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms, 2)

        # Check that the bootstrap file exists
        self.assertTrue(os.path.exists(self.bootstrap_path))

class TestBootstrap_RF_STHR_RMSTID(unittest.TestCase):

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'rf'
        self.alarm_type = 'sthr'
        self.metric = 'rmstid'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        self.bootstrap_path = f"results/{self.device}/{self.dataset_path}/bootstraps"
        self.category_path = f"{self.bootstrap_path}/{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms"

    def test_0_bootstrap_slice_completes(self):
        """Ensure that the bootstrap slice job completes"""
        
        slice_path = f"{self.category_path}/slice_0.pkl"

        # Remove bootstrap file if it exists
        try:
            os.remove(slice_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms, 0)

        # Check that the bootstrap file exists
        self.assertTrue(os.path.exists(self.bootstrap_path))

    def test_1_bootstrap_slice_completes(self):
        """Ensure that the bootstrap slice job completes"""
        
        slice_path = f"{self.category_path}/slice_1.pkl"

        # Remove bootstrap file if it exists
        try:
            os.remove(slice_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms, 1)

        # Check that the bootstrap file exists
        self.assertTrue(os.path.exists(self.bootstrap_path))

    def test_2_bootstrap_slice_completes(self):
        """Ensure that the bootstrap slice job completes"""
        
        slice_path = f"{self.category_path}/slice_2.pkl"

        # Remove bootstrap file if it exists
        try:
            os.remove(slice_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms, 2)

        # Check that the bootstrap file exists
        self.assertTrue(os.path.exists(self.bootstrap_path))

# class TestBootstrap_DCPH_STHR_AUROC(unittest.TestCase):

#     def setUp(self):
#         self.device = TEST_DEVICE
#         self.dataset_path = TEST_DATASET_PATH
#         self.model_type = 'dcph'
#         self.alarm_type = 'sthr'
#         self.metric = 'auroc'
#         self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

#         bootstrap_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_bootstrap"
#         self.bootstrap_path = f"results/{self.device}/{self.dataset_path}/bootstraps/{bootstrap_name}.pkl"

#     def test_a_bootstrap_completes(self):
#         """Ensure that the bootstrap job completes"""
        
#         # Remove bootstrap file if it exists
#         try:
#             os.remove(self.bootstrap_path)
#         except FileNotFoundError:
#             pass

#         # Run the job
#         main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms)

#         # Check that the bootstrap file exists
#         self.assertTrue(os.path.exists(self.bootstrap_path))