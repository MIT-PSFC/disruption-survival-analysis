import unittest

import os

from optuna_job import main
from tests.test_manage_datasets import TEST_DEVICE, TEST_DATASET_PATH, TEST_WARNING_TIME

NUM_TRIALS = 2

class TestTune_RF_STHR_AUROC(unittest.TestCase):
    """Tests for Random Forest Tuning"""

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'rf'
        self.alarm_type = 'sthr'
        self.metric = 'auroc'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        sweep_config_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_sweep"
        self.sweep_config_path = f"{self.device}/{self.dataset_path}/sweeps/{sweep_config_name}.yaml"

    def test_a_study_file_created(self):
        """Ensure that the study file is created after running the job"""
        # Remove the study file if it exists
        study_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_study"
        study_path = f"results/{self.device}/{self.dataset_path}/studies/{study_name}.db"
        try:
            os.remove(study_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.sweep_config_path)

        # Check that the study file was created
        self.assertTrue(os.path.exists(study_path))

class TestTune_RF_STHR_AUWTC(unittest.TestCase):
    """Tests for Random Forest Tuning"""

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'rf'
        self.alarm_type = 'sthr'
        self.metric = 'auwtc'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        sweep_config_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_sweep"
        self.sweep_config_path = f"{self.device}/{self.dataset_path}/sweeps/{sweep_config_name}.yaml"

    def test_a_study_file_created(self):
        """Ensure that the study file is created after running the job"""
        # Remove the study file if it exists
        study_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_study"
        study_path = f"results/{self.device}/{self.dataset_path}/studies/{study_name}.db"
        try:
            os.remove(study_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.sweep_config_path)

        # Check that the study file was created
        self.assertTrue(os.path.exists(study_path))

class TestTune_DCPH_STHR_AUROC(unittest.TestCase):
    """Tests for Random Forest Tuning"""

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'dcph'
        self.alarm_type = 'sthr'
        self.metric = 'auroc'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        sweep_config_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_sweep"
        self.sweep_config_path = f"{self.device}/{self.dataset_path}/sweeps/{sweep_config_name}.yaml"

    def test_a_study_file_created(self):
        """Ensure that the study file is created after running the job"""
        # Remove the study file if it exists
        study_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_study"
        study_path = f"results/{self.device}/{self.dataset_path}/studies/{study_name}.db"
        try:
            os.remove(study_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.sweep_config_path)

        # Check that the study file was created
        self.assertTrue(os.path.exists(study_path))

    # def test_b_ensure_nothing_breaks(self):
    #     """Ensure that nothing breaks when running the job"""
    #     try:
    #         for i in range(NUM_TRIALS):
    #             main(self.sweep_config_path)
    #     except Exception as e:
    #         self.fail(f"Job failed with exception: {e}")