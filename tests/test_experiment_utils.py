import unittest
import numpy as np
import pandas as pd

from tests.test_manage_datasets import TEST_DEVICE, TEST_DATASET_PATH

from disruption_survival_analysis.experiment_utils import label_shot_data, area_under_curve

# Labeling data tests

class TestLabelShotData(unittest.TestCase):
    """Tests for the function label_shot_data"""

    def test_label_shot_data_disruptive(self):
        """Ensure that the label_shot_data function labels disruptive shots correctly
        """

        # Create a Pandas dataframe of shot data with a single shot and only the 'time' column
        shot_data = pd.DataFrame({'time': [0.1, 0.2, 0.3, 0.4, 0.5]})

        # Call the label_shot_data function as a disruptive shot
        labels = label_shot_data(shot_data, True, 0.2)

        # Check that the returned data is the same length as the input data
        self.assertEqual(len(labels), 5)

        # Check that the first three labels are 0 and the last two are 1
        self.assertEqual(labels[0:3].sum(), 0)
        self.assertEqual(labels[3:].sum(), 2)

    def test_label_shot_data_non_disruptive(self):
        """Ensure that the label_shot_data function labels non-disruptive shots correctly
        """

        # Create a Pandas dataframe of shot data with a single shot and only the 'time' column
        shot_data = pd.DataFrame({'time': [0.1, 0.2, 0.3, 0.4, 0.5]})

        # Call the label_shot_data function as a non-disruptive shot
        labels = label_shot_data(shot_data, False, 0.2)

        # Check that the returned data is the same length as the input data
        self.assertEqual(len(labels), 5)

        # Check that the sum of the labels is 0
        self.assertEqual(labels.sum(), 0)

# Evaluation metric tests

class TestTimesliceMicroAverage(unittest.TestCase):
    """Tests for the function timeslice_micro_average"""

class TestAreaUnderCurve(unittest.TestCase):
    """Tests for the function area_under_curve"""

    def test_area_under_curve_straight_line(self):
        # Make a straight line in x and y and ensure the area under the curve is correct

        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 2, 3, 4])

        auc = area_under_curve(x, y)

        self.assertEqual(auc, 8)

    def test_area_under_curve_straight_line_reversed(self):
        # Make a straight line in x and y and ensure the area under the curve is correct

        x = np.array([0, 1, 2, 3, 4])
        y = np.array([4, 3, 2, 1, 0])

        auc = area_under_curve(x, y)

        self.assertEqual(auc, 8)

    def test_area_under_curve_cutoff(self):
        # Ensure that the area under the curve is calculated correctly when a cutoff is specified

        x = np.array([0, 1, 2.1, 3])
        y = np.array([0, 4, 999, 999])

        auc = area_under_curve(x, y, x_cutoff=2)

        # Area should be the area of a triangle with base 1 and height 4, 
        # plus the area of a rectangle with width 1 and height 4
        self.assertEqual(auc, 6)


class TestCalculateF1Scores(unittest.TestCase):
    """Tests for the function calculate_f1_scores"""
