import unittest

import numpy as np

from tests.test_manage_datasets import TEST_DEVICE, TEST_DATASET_PATH, TEST_WARNING_TIME

from disruption_survival_analysis.experiment_utils import load_experiment_config
from disruption_survival_analysis.model_utils import get_model_for_experiment
from disruption_survival_analysis.manage_datasets import load_dataset


# Stuff being tested

from disruption_survival_analysis.DisruptionPredictors import DisruptionPredictor, DisruptionPredictorSM, DisruptionPredictorRF, DisruptionPredictorKM, MAX_FUTURE_LIFETIME

class TestDisruptionPredictor(unittest.TestCase):
    """Tests for the base class DisruptionPredictor"""

    def setUp(self):
        """Set up a DisruptionPredictor instance for testing"""
        
        # Load a config file for a simple RF model
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'rf', 'sthr', 'auroc', 0.01)

        # Get some important information about the model
        trained_required_warning_time = experiment_config["required_warning_time"]
        
        # Create model
        self.model = get_model_for_experiment(experiment_config, "test")

        # Create predictor instance
        self.predictor = DisruptionPredictor("Test Predictor", self.model, trained_required_warning_time)

    def test_get_correct_instance(self):
        """Ensure that the correct class instance is returned"""

        if not isinstance(self.predictor, DisruptionPredictor):
            self.fail("DisruptionPredictor instance not returned")

class TestDisruptionPredictorDCPH(unittest.TestCase):
    """Tests for Survival Model DCPH DisruptionPredictor"""

    def setUp(self):

        # Load a config file for a DCPH model
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'dcph', 'sthr', 'auroc', TEST_WARNING_TIME)

        # Get some important information about the model
        trained_required_warning_time = experiment_config["required_warning_time"]
        trained_horizon = experiment_config["hyperparameters"]["horizon"]
        
        # Create model
        self.model = get_model_for_experiment(experiment_config, "test")

        # Create predictor instance
        self.predictor = DisruptionPredictorSM("Test Predictor", self.model, trained_required_warning_time, trained_horizon)

        # Load some test data
        self.test_data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, "val")

    def test_rmst_not_zero(self):
        """Ensure that the RMST is not always very low for a given shot"""

        # Get a shot number from the data
        shot = self.test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = self.test_data[self.test_data["shot"] == shot]

        # Get the RMST for that shot
        rmst = self.predictor.get_rmst(shot_data)

        # Check that the RMST is not always very low for a given shot
        if np.allclose(rmst, 0, atol=0.001):
            self.fail("RMST always very low for a given shot")


# class TestDisruptionPredictorRF(unittest.TestCase):
#     """Tests for Random Forest DisruptionPredictor"""

#     def test_get_correct_instance(self):
#         """Ensure that the correct class instance is returned"""

#         self.fail()

#     def test_calculate_risk_at_time_shape(self):
#         """Ensure that the function calculate_risk_at_time() returns a Pandas
#         DataFrame with two columns and the same number of rows as the input"""

#         self.fail()

#     def test_calculate_risk_at_time_columns(self):
#         """Ensure that the function calculate_risk_at_time() returns a Pandas
#         DataFrame with the correct columns"""

#         self.fail()

#     def test_calculate_risk_at_time_no_modify(self):
#         """Ensure that the function calculate_risk_at_time() does not modify the input"""

#         self.fail()

#     def test_calculate_ettd_at_time(self):
#         """Ensure that the function calculate_ettd_at_time() returns a Pandas
#         DataFrame with two columns and the same number of rows as the input"""

#         self.fail()

#     def test_calculate_ettd_at_time_columns(self):
#         """Ensure that the function calculate_ettd_at_time() returns a Pandas
#         DataFrame with the correct columns"""

#         self.fail()

#     def test_calculate_ettd_at_time_no_modify(self):
#         """Ensure that the function calculate_ettd_at_time() does not modify the input"""

#         self.fail()

class TestDisruptionPredictorKM(unittest.TestCase):
    """Tests for Kaplan-Meier DisruptionPredictor"""

    def setUp(self):
        """Set up a DisruptionPredictor instance for testing"""
        
        # Load a config file for a simple RF model
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'km', 'sthr', 'auroc', 0.01)

        # Get some important information about the model
        trained_required_warning_time = experiment_config["required_warning_time"]
        trained_class_time = experiment_config["hyperparameters"]["class_time"]
        trained_fit_time = experiment_config["hyperparameters"]["fit_time"]
        trained_horizon = experiment_config["hyperparameters"]["horizon"]
        
        # Create model
        self.model = get_model_for_experiment(experiment_config, "test")

        # Create predictor instance
        self.predictor = DisruptionPredictorKM("Test Predictor", 
                                               self.model, 
                                               trained_required_warning_time,
                                               trained_class_time,
                                               trained_fit_time,
                                               trained_horizon)

    def test_get_risks(self):
        """Ensure the get_risks() returns a numpy array with the correct columns"""

        # Load some test data
        test_data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, "test")
        # Get a shot number from the data
        shot = test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = test_data[test_data["shot"] == shot]

        try:
            # Get the risk for that shot
            risks = self.predictor.get_risks(shot_data)
        except Exception as e:
            self.fail("get_risks() raised an exception: " + str(e))
        
        # Check the shape of risks
        if risks.shape[0] != shot_data.shape[0]:
            self.fail("Number of rows in risks does not match input")

    def test_get_rmst(self):
        """ Ensure that get_rmst() returns a numpy array with the correct shape"""

        # Load some test data
        test_data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, "test")
        # Get a shot number from the data
        shot = test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = test_data[test_data["shot"] == shot]

        try:
            # Get the risk for that shot
            rmst = self.predictor.get_rmst(shot_data)
        except Exception as e:
            self.fail("get_rmst() raised an exception: " + str(e))
        
        # Check the shape of rmst
        if rmst.shape[0] != shot_data.shape[0]:
            self.fail("Number of columns in rmst does not match input")

    def test_risk_outside_range(self):
        """In the shot before there are enough data points to fill the fitting window, ensure that the risk is 0"""

        # Load some test data
        test_data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, "test")
        # Get a shot number from the data
        shot = test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = test_data[test_data["shot"] == shot]
        # Get the times in this shot data
        times = shot_data["time"].to_numpy()

        risks = self.predictor.get_risks(shot_data)

        # Check that the risk is 0 for all times before the fitting window
        for i in range(len(times)):
            if times[i] < self.predictor.trained_fit_time:
                if risks[i] != 0:
                    self.fail("Risk not 0 before fitting window")
            # Check that the risk is not 0 for all times after the fitting window
            else:
                if risks[i] == 0:
                    self.fail("Risk 0 after fitting window")

    def test_rmst_outside_range(self):
        """In the shot before there are enough data points to fill the fitting window, ensure that the rmst is np.nan"""

        # Load some test data
        test_data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, "test")
        # Get a shot number from the data
        shot = test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = test_data[test_data["shot"] == shot]
        # Get the times in this shot data
        times = shot_data["time"].to_numpy()

        rmst = self.predictor.get_rmst(shot_data)

        # Check that the rmst is maxxed for all times before the fitting window
        for i in range(len(times)):
            if times[i] < self.predictor.trained_fit_time:
                if not np.isclose(rmst[i], MAX_FUTURE_LIFETIME, atol=0.001):
                    self.fail(f"RMST not maxxed before fitting window, time = {times[i]}")