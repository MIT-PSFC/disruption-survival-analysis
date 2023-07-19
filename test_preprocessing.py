"""Test functions to ensure that the data preprocessing works as expected
"""
import unittest

from manage_datasets import *

TEST_DEVICE = 'cmod'
TEST_DATASET = '5k_random_256_shots_60%_flattop'
TEST_DATASET_TRAIN = TEST_DATASET + '_train'
TEST_DATASET_TEST = TEST_DATASET + '_test'
TEST_DATASET_VAL = TEST_DATASET + '_val'

class TestLoadDataset(unittest.TestCase):

    def test_no_negative_time(self):
        """Ensure there is no negative time in the datasets
        """

        self.data = load_dataset(TEST_DEVICE, TEST_DATASET_TRAIN)
        self.assertTrue((self.data['time'] >= 0).all())

        self.data = load_dataset(TEST_DEVICE, TEST_DATASET_TEST)
        self.assertTrue((self.data['time'] >= 0).all())

        self.data = load_dataset(TEST_DEVICE, TEST_DATASET_VAL)
        self.assertTrue((self.data['time'] >= 0).all())

    def test_no_negative_disruption_time(self):
        """
        Ensure the time_until_disrupt is not negative
        """

        for dataset in [TEST_DATASET_TRAIN, TEST_DATASET_TEST, TEST_DATASET_VAL]:
            self.data = load_dataset('cmod', dataset)
            # Remove NaN slices (don't care about that for this test)
            self.data = self.data[~self.data['time_until_disrupt'].isnull()]
            self.assertTrue((self.data['time_until_disrupt'] >= 0).all())


class TestLoadFeaturesOutcomes(unittest.TestCase):

    def test_no_negative_outcome_time_synthetic(self):
        """Ensure there are no negative time to event in the outcomes
        with synthetic data
        """

        self.features, self.outcomes = load_features_outcomes('synthetic', 'test', features=['ip','feat2'])
        
        # Assert that the times have actually been updated
        data = load_dataset('synthetic', 'test')
        self.assertTrue((self.outcomes['time'] != data['time']).any())

        # Assert that there are no negative times
        self.assertTrue((self.outcomes['time'] >= 0).all())

    def test_no_negative_outcome_time_real(self):
        """Ensure there are no negative time to event in the outcomes
        with real data
        """

        self.features, self.outcomes = load_features_outcomes('cmod', 'random100_train')
        self.assertTrue((self.outcomes['time'] >= 0).all())

        self.features, self.outcomes = load_features_outcomes('cmod', 'random100_test')
        self.assertTrue((self.outcomes['time'] >= 0).all())

        self.features, self.outcomes = load_features_outcomes('cmod', 'random100_val')
        self.assertTrue((self.outcomes['time'] >= 0).all())

    def test_no_zero_outcome_real(self):
        """Ensure there are no zero time to event in the outcomes"""

        _, self.outcomes = load_features_outcomes('cmod', 'random100_train')
        self.assertTrue((self.outcomes['time'] != 0).all())

        _, self.outcomes = load_features_outcomes('cmod', 'random100_test')
        self.assertTrue((self.outcomes['time'] != 0).all())

        _, self.outcomes = load_features_outcomes('cmod', 'random100_val')
        self.assertTrue((self.outcomes['time'] != 0).all())

    def test_binary_outcomes(self):
        """Ensure that the outcomes are binary and not all 1's or all 0's"""

        _, self.outcomes = load_features_outcomes(TEST_DEVICE, TEST_DATASET_TRAIN)
        self.assertTrue((self.outcomes['event'] == 0).any())
        self.assertTrue((self.outcomes['event'] == 1).any())

        _, self.outcomes = load_features_outcomes(TEST_DEVICE, TEST_DATASET_TEST)
        self.assertTrue((self.outcomes['event'] == 0).any())
        self.assertTrue((self.outcomes['event'] == 1).any())

        _, self.outcomes = load_features_outcomes(TEST_DEVICE, TEST_DATASET_VAL)
        self.assertTrue((self.outcomes['event'] == 0).any())
        self.assertTrue((self.outcomes['event'] == 1).any())

class TestShotLists(unittest.TestCase):

    def test_all_shots_covered(self):
        """Ensure that all shots are covered by the disruptive and non-disruptive shot methods
        """

        # Set the dataset
        device = 'synthetic'
        dataset = 'synthetic100_train'

        # Get the list of shots
        shots = get_shot_list(device, dataset)

        # Get the list of disruptive shots
        disruptive_shots = get_disruptive_shot_list(device, dataset)
        non_disruptive_shots = get_non_disruptive_shot_list(device, dataset)

        # Ensure that all shots are covered
        for shot in shots:
            self.assertTrue(shot in disruptive_shots or shot in non_disruptive_shots)

        # Ensure separation
        for shot in disruptive_shots:
            self.assertTrue(shot not in non_disruptive_shots)