"""Test functions to ensure that the data preprocessing works as expected
"""
import unittest

from preprocess_datasets import load_features_outcomes, load_dataset

class TestLoadDataset(unittest.TestCase):

    def test_no_negative_time(self):
        """Ensure there is no negative time in the datasets
        """

        self.data = load_dataset('cmod', 'random100_train')
        self.assertTrue((self.data['time'] >= 0).all())

        self.data = load_dataset('cmod', 'random100_test')
        self.assertTrue((self.data['time'] >= 0).all())

        self.data = load_dataset('cmod', 'random100_val')
        self.assertTrue((self.data['time'] >= 0).all())

    def test_no_negative_disruption_time(self):
        """
        Ensure the time_until_disrupt is not negative
        """

        for dataset in ['random100_train', 'random100_test', 'random100_val']:
            self.data = load_dataset('cmod', dataset)
            # Remove NaN slices (don't care about that for this test)
            self.data = self.data[~self.data['time_until_disrupt'].isnull()]
            self.assertTrue((self.data['time_until_disrupt'] >= 0).all())


class TestLoadFeaturesOutcomes(unittest.TestCase):

    def test_no_negative_outcome_time_synthetic(self):
        """Ensure there are no negative time to event in the outcomes
        with synthetic data
        """

        self.features, self.outcomes = load_features_outcomes('synthetic', 'test_train', features=['ip','feat2'])
        
        # Assert that the times have actually been updated
        data = load_dataset('synthetic', 'test_train')
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

    

