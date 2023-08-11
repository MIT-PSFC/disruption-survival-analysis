"""Test functions to ensure that the data management works as expected
"""
import os
import time
import unittest

from manage_datasets import *

TEST_DEVICE = 'cmod'
TEST_DATASET_PATH = 'no_ufo_flattop_7736_shots_6%_disruptive'

class TestLoadDataset(unittest.TestCase):

    def test_no_negative_time(self):
        """Ensure there is no negative time in the experiment datasets
        """

        for category in ['train', 'test', 'val']:
            data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, category)
            self.assertTrue((data['time'] >= 0).all())

    def test_no_negative_disruption_time(self):
        """
        Ensure the time_until_disrupt is not negative
        """

        for category in ['train', 'test', 'val']:
            data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, category)
            # Remove NaN slices (don't care about that for this test)
            data = data[~data['time_until_disrupt'].isnull()]
            self.assertTrue((data['time_until_disrupt'] >= 0).all())

    def test_sort_orders(self):
        """Ensure that the data is sorted by shot number and time
        """

        for category in ['train', 'test', 'val']:
            data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, category)

            # Check that the data is sorted by shot number
            # Get the difference between each shot number and the previous one
            diff_shot = data['shot'].diff()
            # The first shot will have a NaN difference, so we need to remove that
            diff_shot = diff_shot.tail(-1)
            # Check that the remaining differences are always positive
            self.assertTrue((diff_shot >= 0).all())

            # For each shot, check that the data is sorted by time
            for shot in data['shot'].unique():
                shot_data = data[data['shot'] == shot]
                # Get the difference between each time and the previous one
                diff_time = shot_data['time'].diff()
                # The first time will have a NaN difference, so we need to remove that
                diff_time = diff_time.tail(-1)
                # Check that the remaining differences are always positive
                self.assertTrue((diff_time >= 0).all())


class TestLoadFeaturesOutcomes(unittest.TestCase):

    def test_updated_time(self):
        """Ensure that the time to event in outcomes is different from the regular input time 
        """
        for category in ['train', 'test', 'val']:
            _, outcomes = load_features_outcomes(TEST_DEVICE, TEST_DATASET_PATH, category)

            data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, category)
            
            self.assertTrue((outcomes['time'] != data['time']).any())

    def test_no_negative_outcome_time(self):
        """Ensure there are no negative time to event in the outcomes
        """

        for category in ['train', 'test', 'val']:
            _, outcomes = load_features_outcomes(TEST_DEVICE, TEST_DATASET_PATH, category)
            
            # Assert that there are no negative times
            self.assertTrue((outcomes['time'] >= 0).all())

    def test_no_zero_outcome_time(self):
        """Ensure there are no zero time to event in the outcomes"""

        for category in ['train', 'test', 'val']:
            _, outcomes = load_features_outcomes(TEST_DEVICE, TEST_DATASET_PATH, category)
            
            # Assert that there are no zero times
            self.assertTrue((outcomes['time'] != 0).all())
        
    def test_binary_outcomes(self):
        """Ensure that the outcomes are binary and not all 1's or all 0's"""

        for category in ['train', 'test', 'val']:
            _, outcomes = load_features_outcomes(TEST_DEVICE, TEST_DATASET_PATH, category)
            
            # Assert that there are both 0's and 1's
            self.assertTrue((outcomes['event'] == 0).any())
            self.assertTrue((outcomes['event'] == 1).any())

class TestShotLists(unittest.TestCase):

    def test_all_shots_covered(self):
        """Ensure that all shots are covered by the disruptive and non-disruptive shot loading functions
        """

        for category in ['train', 'test', 'val']:
            # Get the list of shots
            shot_list = load_shot_list(TEST_DEVICE, TEST_DATASET_PATH, category)

            # Get the lists of disruptive and non-disruptive shots
            disruptive_shot_list = load_disruptive_shot_list(TEST_DEVICE, TEST_DATASET_PATH, category)
            non_disruptive_shot_list = load_non_disruptive_shot_list(TEST_DEVICE, TEST_DATASET_PATH, category)

            # Ensure that all shots are covered
            for shot in shot_list:
                self.assertTrue(shot in disruptive_shot_list or shot in non_disruptive_shot_list)


    def test_no_shot_overlap(self):
        """Ensure that there is no overlap between the disruptive and non-disruptive shot lists
        """

        for category in ['train', 'test', 'val']:
            disruptive_shot_list = load_disruptive_shot_list(TEST_DEVICE, TEST_DATASET_PATH, category)
            non_disruptive_shot_list = load_non_disruptive_shot_list(TEST_DEVICE, TEST_DATASET_PATH, category)

            for shot in disruptive_shot_list:
                self.assertTrue(shot not in non_disruptive_shot_list)

class TestCreatedTrainingSets(unittest.TestCase):

    def test_make_training_sets(self):
        """Ensure that the training sets are created without error
        """

        try:
            make_training_sets(TEST_DEVICE, TEST_DATASET_PATH)
        except:
            self.fail("make_training_sets() raised an exception unexpectedly!")

        # Ensure that the training sets that exist are actually new
        for category in ['train', 'test', 'val']:
            path = f'data/{TEST_DEVICE}/{TEST_DATASET_PATH}/{category}.csv'
            self.assertTrue(os.path.exists(path))


    def test_no_null_values(self):
        """Ensure there are no null values in the training sets"""

        for category in ['train', 'test', 'val']:
            data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, category)
            # Ignore the time_until_disrupt column, since it is allowed to be null
            data = data.drop(columns=['time_until_disrupt'])

            self.assertFalse(data.isnull().values.any())

    def test_no_short_shots(self):
        """Ensure there are no shots with less than 10 slices"""

        for category in ['train', 'test', 'val']:
            data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, category)
            shots = data.groupby('shot').size()

            self.assertTrue((shots >= 10).all())

    def test_no_dropped_features(self):
        """Ensure that the dropped features are actually gone"""

        for category in ['train', 'test', 'val']:
            data = load_dataset(TEST_DEVICE, TEST_DATASET_PATH, category)
            # Ensure that the dropped features are not in the data
            for feature in DROPPED_FEATURES:
                self.assertTrue(feature not in data.columns)

    def test_no_category_imbalance(self):
        """Ensure that there is a roughly even number of disruptive shots in the training sets,
        proportional to their total size"""
        
        ratio = {}
        ratio_epsilon = 0.02

        for category in ['train', 'test', 'val']:
            num_disrupt = len(load_disruptive_shot_list(TEST_DEVICE, TEST_DATASET_PATH, category))
            total_size = len(load_shot_list(TEST_DEVICE, TEST_DATASET_PATH, category))

            ratio[category] = num_disrupt / total_size
        
        # Ensure that the ratios are roughly equal
        self.assertTrue(abs(ratio['train'] - ratio['test']) < ratio_epsilon)
        self.assertTrue(abs(ratio['train'] - ratio['val']) < ratio_epsilon)
        self.assertTrue(abs(ratio['test'] - ratio['val']) < ratio_epsilon)

        