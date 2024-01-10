import unittest

import os
import dill

from training_job import main
from tests.test_manage_datasets import TEST_DEVICE, TEST_DATASET_PATH, TEST_WARNING_TIME
from disruption_survival_analysis.experiment_utils import load_experiment_config

class TestTrain_RF_STHR_AUROC(unittest.TestCase):

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'rf'
        self.alarm_type = 'sthr'
        self.metric = 'auroc'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        model_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms"
        self.model_path = f"results/{self.device}/{self.dataset_path}/models/{model_name}.pkl"
        self.config_path = f"results/{self.device}/{self.dataset_path}/configs/{model_name}.yaml"
        experiment_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_experiment"
        self.experiment_path = f"results/{self.device}/{self.dataset_path}/experiments/{experiment_name}.pkl"

    def test_a_training_completes(self):
        """Ensure that the training job completes"""
        
        # Remove experiment, model and config files if they exist
       
        try:
            os.remove(self.model_path)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.config_path)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.experiment_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms)

        # Check that the experiment, model, and config files were created
        self.assertTrue(os.path.exists(self.experiment_path))
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.config_path))

    def test_b_experiment_values(self):
        # Check that the experiment has the correct values
        with open(self.experiment_path, 'rb') as f:
            experiment = dill.load(f)

        self.assertEqual(experiment.device, self.device)
        self.assertEqual(experiment.dataset_path, self.dataset_path)
        self.assertEqual(experiment.model_type, self.model_type)
        self.assertEqual(experiment.alarm_type, self.alarm_type)

    def test_c_predictor_values(self):
        # Check that the predictor values match the config

        required_warning_time = float(self.required_warning_time_ms) / 1000
        config = load_experiment_config(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, required_warning_time)

        with open(self.experiment_path, 'rb') as f:
            experiment = dill.load(f)

        predictor = experiment.predictor

        self.assertEqual(predictor.trained_required_warning_time, config["required_warning_time"])
        self.assertEqual(predictor.trained_class_time, config["hyperparameters"]["class_time"])


class TestTrain_RF_STHR_AUWTC(unittest.TestCase):

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'rf'
        self.alarm_type = 'sthr'
        self.metric = 'auwtc'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        model_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms"
        self.model_path = f"results/{self.device}/{self.dataset_path}/models/{model_name}.pkl"
        self.config_path = f"results/{self.device}/{self.dataset_path}/configs/{model_name}.yaml"
        experiment_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_experiment"
        self.experiment_path = f"results/{self.device}/{self.dataset_path}/experiments/{experiment_name}.pkl"

    def test_a_training_completes(self):
        """Ensure that the training job completes"""
        
        # Remove experiment, model and config files if they exist
       
        try:
            os.remove(self.model_path)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.config_path)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.experiment_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms)

        # Check that the experiment, model, and config files were created
        self.assertTrue(os.path.exists(self.experiment_path))
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.config_path))

    def test_b_experiment_values(self):
        # Check that the experiment has the correct values
        with open(self.experiment_path, 'rb') as f:
            experiment = dill.load(f)

        self.assertEqual(experiment.device, self.device)
        self.assertEqual(experiment.dataset_path, self.dataset_path)
        self.assertEqual(experiment.model_type, self.model_type)
        self.assertEqual(experiment.alarm_type, self.alarm_type)

    def test_c_predictor_values(self):
        # Check that the predictor values match the config

        required_warning_time = float(self.required_warning_time_ms) / 1000
        config = load_experiment_config(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, required_warning_time)

        with open(self.experiment_path, 'rb') as f:
            experiment = dill.load(f)

        predictor = experiment.predictor

        self.assertEqual(predictor.trained_required_warning_time, config["required_warning_time"])
        self.assertEqual(predictor.trained_class_time, config["hyperparameters"]["class_time"])

class TestTrain_RF_STHR_RMSTID(unittest.TestCase):

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'rf'
        self.alarm_type = 'sthr'
        self.metric = 'rmstid'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        model_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms"
        self.model_path = f"results/{self.device}/{self.dataset_path}/models/{model_name}.pkl"
        self.config_path = f"results/{self.device}/{self.dataset_path}/configs/{model_name}.yaml"
        experiment_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_experiment"
        self.experiment_path = f"results/{self.device}/{self.dataset_path}/experiments/{experiment_name}.pkl"

    def test_a_training_completes(self):
        """Ensure that the training job completes"""
        
        # Remove experiment, model and config files if they exist
       
        try:
            os.remove(self.model_path)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.config_path)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.experiment_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms)

        # Check that the experiment, model, and config files were created
        self.assertTrue(os.path.exists(self.experiment_path))
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.config_path))

    def test_b_experiment_values(self):
        # Check that the experiment has the correct values
        with open(self.experiment_path, 'rb') as f:
            experiment = dill.load(f)

        self.assertEqual(experiment.device, self.device)
        self.assertEqual(experiment.dataset_path, self.dataset_path)
        self.assertEqual(experiment.model_type, self.model_type)
        self.assertEqual(experiment.alarm_type, self.alarm_type)

    def test_c_predictor_values(self):
        # Check that the predictor values match the config

        required_warning_time = float(self.required_warning_time_ms) / 1000
        config = load_experiment_config(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, required_warning_time)

        with open(self.experiment_path, 'rb') as f:
            experiment = dill.load(f)

        predictor = experiment.predictor

        self.assertEqual(predictor.trained_required_warning_time, config["required_warning_time"])
        self.assertEqual(predictor.trained_class_time, config["hyperparameters"]["class_time"])

class TestTrain_DCPH_STHR_AUROC(unittest.TestCase):

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'dcph'
        self.alarm_type = 'sthr'
        self.metric = 'auroc'
        self.required_warning_time_ms = str(int(TEST_WARNING_TIME*1000))

        model_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms"
        self.model_path = f"results/{self.device}/{self.dataset_path}/models/{model_name}.pkl"
        self.config_path = f"results/{self.device}/{self.dataset_path}/configs/{model_name}.yaml"
        experiment_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_experiment"
        self.experiment_path = f"results/{self.device}/{self.dataset_path}/experiments/{experiment_name}.pkl"

    def test_a_training_completes(self):
        """Ensure that the training job completes"""
        
        # Remove experiment, model and config files if they exist
       
        try:
            os.remove(self.model_path)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.config_path)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.experiment_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms)

        # Check that the experiment, model, and config files were created
        self.assertTrue(os.path.exists(self.experiment_path))
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.config_path))

    def test_b_experiment_values(self):
        # Check that the experiment has the correct values
        with open(self.experiment_path, 'rb') as f:
            experiment = dill.load(f)

        self.assertEqual(experiment.device, self.device)
        self.assertEqual(experiment.dataset_path, self.dataset_path)
        self.assertEqual(experiment.model_type, self.model_type)
        self.assertEqual(experiment.alarm_type, self.alarm_type)

    def test_c_predictor_values(self):
        # Check that the predictor values match the config

        required_warning_time = float(self.required_warning_time_ms) / 1000
        config = load_experiment_config(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, required_warning_time)

        with open(self.experiment_path, 'rb') as f:
            experiment = dill.load(f)

        predictor = experiment.predictor

        self.assertEqual(predictor.trained_required_warning_time, config["required_warning_time"])
        self.assertEqual(predictor.trained_horizon, config["hyperparameters"]["horizon"])


class TestTrain_NO_STUDY(unittest.TestCase):
    # Ensure that training job can run from just a config, no study

    def setUp(self):
        self.device = TEST_DEVICE
        self.dataset_path = TEST_DATASET_PATH
        self.model_type = 'rf'
        self.alarm_type = 'sthr'
        self.metric = 'auroc'
        self.required_warning_time_ms = 30

        model_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms"
        self.model_path = f"results/{self.device}/{self.dataset_path}/models/{model_name}.pkl"
        self.config_path = f"results/{self.device}/{self.dataset_path}/configs/{model_name}.yaml"
        experiment_name = f"{self.model_type}_{self.alarm_type}_{self.metric}_{self.required_warning_time_ms}ms_experiment"
        self.experiment_path = f"results/{self.device}/{self.dataset_path}/experiments/{experiment_name}.pkl"

    def test_a_training_completes(self):
        """Ensure that the training job completes"""
        
        # Remove experiment, model and config files if they exist
       
        try:
            os.remove(self.model_path)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.experiment_path)
        except FileNotFoundError:
            pass

        # Run the job
        main(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, self.required_warning_time_ms)

        # Check that the experiment, model, and config files were created
        self.assertTrue(os.path.exists(self.experiment_path))
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.config_path))

    def test_b_experiment_values(self):
        # Check that the experiment has the correct values
        with open(self.experiment_path, 'rb') as f:
            experiment = dill.load(f)

        self.assertEqual(experiment.device, self.device)
        self.assertEqual(experiment.dataset_path, self.dataset_path)
        self.assertEqual(experiment.model_type, self.model_type)
        self.assertEqual(experiment.alarm_type, self.alarm_type)

    def test_c_predictor_values(self):
        # Check that the predictor values match the config

        required_warning_time = float(self.required_warning_time_ms) / 1000
        config = load_experiment_config(self.device, self.dataset_path, self.model_type, self.alarm_type, self.metric, required_warning_time)

        with open(self.experiment_path, 'rb') as f:
            experiment = dill.load(f)

        predictor = experiment.predictor

        self.assertEqual(predictor.trained_required_warning_time, config["required_warning_time"])
        self.assertEqual(predictor.trained_class_time, config["hyperparameters"]["class_time"])