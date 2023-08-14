import unittest

from disruption_survival_analysis.DisruptionPredictors import *

class TestDisruptionPredictor(unittest.TestCase):
    """Tests for the base class DisruptionPredictor"""

    def test_get_correct_instance(self):
        """Ensure that the correct class instance is returned"""

        self.fail()

    def test_get_all_features(self):
        """Ensure that the correct features are filled in"""

        self.fail()

    def test_removed_features(self):
        """Ensure that the 'shot', 'time', and 'time_until_disrupt' columns are removed"""

        self.fail()

    def test_no_calculate_risk_at_time(self):
        """Ensure that the function calculate_risk_at_time() is not implemented"""

        self.fail()

    def test_no_calculate_ettd_at_time(self):
        """Ensure that the function calculate_ettd_at_time() is not implemented"""

        self.fail()

class TestDisruptionPredictorSM(unittest.TestCase):
    """Tests for Survival Model DisruptionPredictor"""

    def test_get_correct_instance(self):
        """Ensure that the correct class instance is returned"""

        self.fail()

    def test_calculate_risk_at_time_shape(self):
        """Ensure that the function calculate_risk_at_time() returns a Pandas
        DataFrame with two columns and the same number of rows as the input"""

        self.fail()

    def test_calculate_risk_at_time_columns(self):
        """Ensure that the function calculate_risk_at_time() returns a Pandas
        DataFrame with the correct columns"""

        self.fail()

    def test_calculate_risk_at_time_no_modify(self):
        """Ensure that the function calculate_risk_at_time() does not modify the input"""

        self.fail()

    def test_calculate_ettd_at_time(self):
        """Ensure that the function calculate_ettd_at_time() returns a Pandas
        DataFrame with two columns and the same number of rows as the input"""

        self.fail()

    def test_calculate_ettd_at_time_columns(self):
        """Ensure that the function calculate_ettd_at_time() returns a Pandas
        DataFrame with the correct columns"""

        self.fail()

    def test_calculate_ettd_at_time_no_modify(self):
        """Ensure that the function calculate_ettd_at_time() does not modify the input"""

        self.fail()

class TestDisruptionPredictorRF(unittest.TestCase):
    """Tests for Random Forest DisruptionPredictor"""

    def test_get_correct_instance(self):
        """Ensure that the correct class instance is returned"""

        self.fail()

    def test_calculate_risk_at_time_shape(self):
        """Ensure that the function calculate_risk_at_time() returns a Pandas
        DataFrame with two columns and the same number of rows as the input"""

        self.fail()

    def test_calculate_risk_at_time_columns(self):
        """Ensure that the function calculate_risk_at_time() returns a Pandas
        DataFrame with the correct columns"""

        self.fail()

    def test_calculate_risk_at_time_no_modify(self):
        """Ensure that the function calculate_risk_at_time() does not modify the input"""

        self.fail()

    def test_calculate_ettd_at_time(self):
        """Ensure that the function calculate_ettd_at_time() returns a Pandas
        DataFrame with two columns and the same number of rows as the input"""

        self.fail()

    def test_calculate_ettd_at_time_columns(self):
        """Ensure that the function calculate_ettd_at_time() returns a Pandas
        DataFrame with the correct columns"""

        self.fail()

    def test_calculate_ettd_at_time_no_modify(self):
        """Ensure that the function calculate_ettd_at_time() does not modify the input"""

        self.fail()

class TestDisruptionPredictorKM(unittest.TestCase):
    """Tests for Kaplan-Meier DisruptionPredictor"""

    def test_get_correct_instance(self):
        """Ensure that the correct class instance is returned"""

        self.fail()

    def test_calculate_risk_at_time_shape(self):
        """Ensure that the function calculate_risk_at_time() returns a Pandas
        DataFrame with two columns and the same number of rows as the input"""

        self.fail()

    def test_calculate_risk_at_time_columns(self):
        """Ensure that the function calculate_risk_at_time() returns a Pandas
        DataFrame with the correct columns"""

        self.fail()

    def test_calculate_risk_at_time_no_modify(self):
        """Ensure that the function calculate_risk_at_time() does not modify the input"""

        self.fail()

    def test_calculate_ettd_at_time(self):
        """Ensure that the function calculate_ettd_at_time() returns a Pandas
        DataFrame with two columns and the same number of rows as the input"""

        self.fail()

    def test_calculate_ettd_at_time_columns(self):
        """Ensure that the function calculate_ettd_at_time() returns a Pandas
        DataFrame with the correct columns"""

        self.fail()

    def test_calculate_ettd_at_time_no_modify(self):
        """Ensure that the function calculate_ettd_at_time() does not modify the input"""

        self.fail()

    def test_linear_slope(self):
        """Ensure that the function linear_slope() returns something resembling the correct slope"""

        self.fail()