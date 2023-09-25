import unittest

from test_manage_datasets import TEST_DEVICE, TEST_DATASET_PATH

from disruption_survival_analysis.experiment_utils import load_experiment_config
from disruption_survival_analysis.model_utils import get_model_for_experiment
from disruption_survival_analysis.manage_datasets import load_dataset


# Stuff being tested

from disruption_survival_analysis.DisruptionPredictors import DisruptionPredictor, DisruptionPredictorSM, DisruptionPredictorRF, DisruptionPredictorKM

class TestDisruptionPredictor(unittest.TestCase):
    """Tests for the base class DisruptionPredictor"""

    def setUp(self):
        """Set up a DisruptionPredictor instance for testing"""
        
        # Load a config file for a simple RF model
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'rf', 'sthr', 'auroc', 0.02)

        # Get some important information about the model
        trained_required_warning_time = experiment_config["required_warning_time"]
        trained_class_time = experiment_config["class_time"]
        
        # Create model
        self.model = get_model_for_experiment(experiment_config, "test")

        # Create predictor instance
        self.predictor = DisruptionPredictor("Test Predictor", self.model, trained_required_warning_time, trained_class_time)

    def test_get_correct_instance(self):
        """Ensure that the correct class instance is returned"""

        if not isinstance(self.predictor, DisruptionPredictor):
            self.fail("DisruptionPredictor instance not returned")

    def test_get_all_features_none_start(self):
        """Ensure that the features start as None"""

        if self.predictor.features is not None:
            self.fail("Features not None on predictor initialization")

    def test_no_get_risk_at_times(self):
        """Ensure that the function get_risk_at_times() is not implemented"""

        try:
            self.predictor.get_risk_at_times(1, 0, 0)
            self.fail("get_risk_at_times() implemented by the base class!")
        except NotImplementedError:
            pass
        except:
            self.fail("Unexpected error raised by get_risk_at_times()")

    def test_no_get_ettd_at_times(self):
        """Ensure that the function calculate_ettd_at_time() is not implemented"""

        try:
            self.predictor.get_ettd_at_times(1, 0)
            self.fail("get_ettd_at_times() implemented by the base class!")
        except NotImplementedError:
            pass
        except:
            self.fail("Unexpected error raised by get_ettd_at_times()")

class TestDisruptionPredictorSM(unittest.TestCase):
    """Tests for Survival Model DisruptionPredictor"""

    def setUp(self):

        # Load a config file for a simple RF model
        experiment_config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, 'rf', 'sthr', 'auroc', 0.02)

        # Get some important information about the model
        trained_required_warning_time = experiment_config["required_warning_time"]
        trained_horizon = experiment_config["horizon"]
        
        # Create model
        self.model = get_model_for_experiment(experiment_config, "test")

        # Create predictor instance
        self.predictor = DisruptionPredictorSM("Test Predictor", self.model, trained_required_warning_time, trained_horizon)

        # Load some test data
        self.test_data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, "val")

    def test_get_correct_instance(self):
        """Ensure that the correct class instance is returned"""

        if not isinstance(self.predictor, DisruptionPredictorSM):
            self.fail("DisruptionPredictorSM instance not returned")

    def test_feature_fill_risk(self):
        """Ensure that the features are empty at the start and filled after predicting risk"""

        if self.predictor.features is not None:
            self.fail("Features not None on predictor initialization")
        
        # Get a shot number from the data
        shot = self.test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = self.test_data[self.test_data["shot"] == shot]

        # Get the risk for that shot
        self.predictor.get_risk(shot, shot_data)

        if self.predictor.features is None:
            self.fail("Features not filled after predicting risk")

    def test_removed_features(self):
        """Ensure that the 'shot', 'time', and 'time_until_disrupt' columns are removed"""

        # Get a shot number from the data
        shot = self.test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = self.test_data[self.test_data["shot"] == shot]

        # Get the risk for that shot
        self.predictor.get_risk(shot, shot_data)

        # Ensure that the 'shot', 'time', and 'time_until_disrupt' columns are removed
        if "shot" in self.predictor.features:
            self.fail("Shot column not removed from features")
        if "time" in self.predictor.features:
            self.fail("Time column not removed from features")
        if "time_until_disrupt" in self.predictor.features:
            self.fail("Time until disrupt column not removed from features")

    def test_get_risk_at_times_shape(self):
        """Ensure that the function get_risk_at_times() returns a Pandas
        DataFrame with two columns and the same number of rows as the input"""

        # Get a shot number from the data
        shot = self.test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = self.test_data[self.test_data["shot"] == shot]

        # Get the risk for that shot
        risk_at_times = self.predictor.get_risk_at_times(shot, shot_data)

        # Check the shape of risk_at_times
        if risk_at_times.shape[0] != shot_data.shape[0]:
            self.fail("Number of rows in risk_at_times does not match input")
        if risk_at_times.shape[1] != 2:
            self.fail("Number of columns in risk_at_times is not 2")

    def test_get_risk_at_times_columns(self):
        """Ensure that the function get_risk_at_times() returns a Pandas
        DataFrame with the correct columns"""

        # Get a shot number from the data
        shot = self.test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = self.test_data[self.test_data["shot"] == shot]

        # Get the risk for that shot
        risk_at_times = self.predictor.get_risk_at_times(shot, shot_data)

        # Check the columns in risk_at_times
        if "risk" not in risk_at_times.columns:
            self.fail("risk column not present")
        if "time" not in risk_at_times.columns:
            self.fail("time column not present")

    def test_get_risk_at_times_no_modify(self):
        """Ensure that the function get_risk_at_times() does not modify the input"""

        # Get a shot number from the data
        shot = self.test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = self.test_data[self.test_data["shot"] == shot]

        shot_data_copy = shot_data.copy()

        # Get the risk for that shot
        self.predictor.get_risk_at_times(shot, shot_data)

        # Check that the input was not modified
        if not shot_data.equals(shot_data_copy):
            self.fail("Input was modified")
        
    def test_get_ettd_at_times_shape(self):
        """Ensure that the function get_ettd_at_times() returns a Pandas
        DataFrame with two columns and the same number of rows as the input"""

        # Get a shot number from the data
        shot = self.test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = self.test_data[self.test_data["shot"] == shot]

        # Get the expected time to disruption for that shot
        ettd_at_times = self.predictor.get_ettd_at_times(shot, shot_data)

        # Check the shape of ettd_at_times
        if ettd_at_times.shape[0] != shot_data.shape[0]:
            self.fail("Number of rows in ettd_at_times does not match input")
        if ettd_at_times.shape[1] != 2:
            self.fail("Number of columns in ettd_at_times is not 2")

    def test_get_ettd_at_times_columns(self):
        """Ensure that the function get_ettd_at_times() returns a Pandas
        DataFrame with the correct columns"""

        # Get a shot number from the data
        shot = self.test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = self.test_data[self.test_data["shot"] == shot]

        # Get the expected time to disruption for that shot
        ettd_at_times = self.predictor.get_ettd_at_times(shot, shot_data)

        # Check the columns in ettd_at_times
        if "ettd" not in ettd_at_times.columns:
            self.fail("ettd column not present")
        if "time" not in ettd_at_times.columns:
            self.fail("time column not present")

    def test_get_ettd_at_times_no_modify(self):
        """Ensure that the function get_ettd_at_times() does not modify the input"""

        # Get a shot number from the data
        shot = self.test_data.iloc[0]["shot"]
        # Get the data for that shot
        shot_data = self.test_data[self.test_data["shot"] == shot]

        shot_data_copy = shot_data.copy()

        # Get the expected time to disruption for that shot
        ettd_at_times = self.predictor.get_ettd_at_times(shot, shot_data)

        # Check that the input was not modified
        if not shot_data.equals(shot_data_copy):
            self.fail("Input was modified")

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

# class TestDisruptionPredictorKM(unittest.TestCase):
#     """Tests for Kaplan-Meier DisruptionPredictor"""

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

#     def test_linear_slope(self):
#         """Ensure that the function linear_slope() returns something resembling the correct slope"""

#         self.fail()