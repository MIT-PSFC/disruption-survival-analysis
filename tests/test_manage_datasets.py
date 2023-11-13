"""Test functions to ensure that the data management works as expected
"""
import os
import unittest
import numpy as np

from disruption_survival_analysis.manage_datasets import make_training_sets, make_stacked_sets, load_dataset, load_shot_list, load_features_labels, load_features_outcomes, load_disruptive_shot_list, load_non_disruptive_shot_list, load_feature_list, DROPPED_FEATURES

TEST_DEVICE = 'synthetic'
TEST_DATASET_PATH = 'test'

class TestCreatedTrainingSets(unittest.TestCase):
    """Tests for the function make_training_sets()"""

    def test_make_training_sets(self):
        """Ensure that the training sets are created without error
        """

        # Remove training sets if they exist
        for category in ['train', 'test', 'val']:
            path = f'data/{TEST_DEVICE}/{TEST_DATASET_PATH}/{category}.csv'
            if os.path.exists(path):
                os.remove(path)

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
        ratio_epsilon = 0.06

        for category in ['train', 'test', 'val']:
            num_disrupt = len(load_disruptive_shot_list(TEST_DEVICE, TEST_DATASET_PATH, category))
            total_size = len(load_shot_list(TEST_DEVICE, TEST_DATASET_PATH, category))

            ratio[category] = num_disrupt / total_size
        
        # Ensure that the ratios are roughly equal
        if abs(ratio['train'] - ratio['test']) > ratio_epsilon:
            self.fail(f"Class Imbalance! train: {ratio['train']}, test: {ratio['test']}")
        if abs(ratio['train'] - ratio['val']) > ratio_epsilon:
            self.fail(f"Class Imbalance! train: {ratio['train']}, val: {ratio['val']}")
        if abs(ratio['test'] - ratio['val']) > ratio_epsilon:
            self.fail(f"Class Imbalance! test: {ratio['test']}, val: {ratio['val']}")

class TestCreatedStackedSets(unittest.TestCase):
    """Tests for the function make_stacked_sets()"""

    stack_number = 5
    stack_dataset_path = f'{TEST_DATASET_PATH}/stack_{stack_number}'

    def test_make_stacked_sets(self):
        """Ensure that the stacked sets are created without error
        """

        # Remove stacked sets if they exist
        for category in ['train', 'test', 'val']:
            path = f'data/{TEST_DEVICE}/{self.stack_dataset_path}/{category}.csv'
            if os.path.exists(path):
                os.remove(path)

        try:
            for category in ['train', 'test', 'val']:
                make_stacked_sets(TEST_DEVICE, TEST_DATASET_PATH, category, self.stack_number)
        except:
            self.fail("make_stacked_sets() raised an exception unexpectedly!")

        # Ensure that the stacked sets that exist are actually new
        for category in ['train', 'test', 'val']:
            path = f'data/{TEST_DEVICE}/{self.stack_dataset_path}/{category}.csv'
            if not os.path.exists(path):
                self.fail(f"make_stacked_sets() did not create {path}!")

    def test_no_null_values(self):
        """Ensure there are no null values in the stacked sets"""

        for category in ['train', 'test', 'val']:
            data = load_dataset(TEST_DEVICE, self.stack_dataset_path, category)
            # Ignore the time_until_disrupt column, since it is allowed to be null
            data = data.drop(columns=['time_until_disrupt'])

            self.assertFalse(data.isnull().values.any())

    def test_no_short_shots(self):
        """Ensure there are no shots with less than 10 slices"""

        for category in ['train', 'test', 'val']:
            data = load_dataset(TEST_DEVICE, self.stack_dataset_path, category)
            shots = data.groupby('shot').size()

            self.assertTrue((shots >= 10).all())

    def test_no_category_imbalance(self):
        """Ensure that there is a roughly even number of disruptive shots in the training sets,
        proportional to their total size"""
        
        ratio = {}
        ratio_epsilon = 0.06

        for category in ['train', 'test', 'val']:
            num_disrupt = len(load_disruptive_shot_list(TEST_DEVICE, self.stack_dataset_path, category))
            total_size = len(load_shot_list(TEST_DEVICE, self.stack_dataset_path, category))

            ratio[category] = num_disrupt / total_size
        
        # Ensure that the ratios are roughly equal
        if abs(ratio['train'] - ratio['test']) > ratio_epsilon:
            self.fail(f"Class Imbalance! train: {ratio['train']}, test: {ratio['test']}")
        if abs(ratio['train'] - ratio['val']) > ratio_epsilon:
            self.fail(f"Class Imbalance! train: {ratio['train']}, val: {ratio['val']}")
        if abs(ratio['test'] - ratio['val']) > ratio_epsilon:
            self.fail(f"Class Imbalance! test: {ratio['test']}, val: {ratio['val']}")
        
    def test_no_repeat(self):
        """Ensure that the stacked sets actually have unique data in each slice"""

        for category in ['train', 'test', 'val']:
            data = load_dataset(TEST_DEVICE, self.stack_dataset_path, category)

            # Pick the first shot
            shot = data['shot'].unique()[0]
            shot_data = data[data['shot'] == shot]
            # Sort shot data by time
            shot_data = shot_data.sort_values(by='time')

            # Drop the time_until_disrupt, shot, and time columns
            shot_data = shot_data.drop(columns=['time_until_disrupt', 'shot', 'time'])

            # Get the 'base column names' (i.e. the column names without the stack number)
            # Should be the first 8 columns
            base_column_names = shot_data.columns[:8]

            for base_column_name in base_column_names:
                stack_column_names = [f'{base_column_name}_{i}' for i in range(self.stack_number)]

                # Get the stacked values for this column
                stacked_values = shot_data[stack_column_names].values
                # For each row after the first 10, ensure that the values are not the same




class TestLoadDataset(unittest.TestCase):
    """Tests for the function load_dataset()"""

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
    """Tests for the function load_features_outcomes()"""

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

    def test_exact_return(self):
        """Ensure that the function returns the correct features and outcomes dataframes
        for a small dataset"""

        features, outcomes = load_features_outcomes("synthetic", "test", "small")

        # Check that the features are correct
        ip_array = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 4, 4, 10]
        feat2_array = [7]*24

        self.assertTrue((features['ip'] == ip_array).all())
        self.assertTrue((features['feat2'] == feat2_array).all())

        # Check that the outcomes are correct
        event_array = [0]*15 + [1]*9
        # Need to use a small-value epsilon for the time comparison
        time_array = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        self.assertTrue((outcomes['event'] == event_array).all())
        self.assertTrue(np.allclose(outcomes['time'], time_array, atol=5e-4))

    def test_no_non_features(self):
        """Ensure that time, shot, and time_until_disrupt are not in the features"""

        features, _ = load_features_outcomes(TEST_DEVICE, TEST_DATASET_PATH, "train")

        self.assertTrue('time' not in features.columns)
        self.assertTrue('shot' not in features.columns)
        self.assertTrue('time_until_disrupt' not in features.columns)

class TestLoadFeaturesLabels(unittest.TestCase):
    """Tests for the function load_features_labels()"""

    def test_exact_return(self):
        """Ensure that the function returns the correct features and labels
        for a small dataset"""

        features, labels = load_features_labels("synthetic", "test", "small", 5)

        # Check that the features are correct
        ip_array = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 4, 4, 10]
        feat2_array = [7]*24

        self.assertTrue((features['ip'] == ip_array).all())
        self.assertTrue((features['feat2'] == feat2_array).all())

        # Check that the labels are correct
        label_array = [0]*15 + [0]*4 + [1]*5
        self.assertTrue((labels == label_array).all())

    def test_no_non_features(self):
        """Ensure that time, shot, and time_until_disrupt are not in the features"""

        features, _ = load_features_labels(TEST_DEVICE, TEST_DATASET_PATH, "train", 5)

        self.assertTrue('time' not in features.columns)
        self.assertTrue('shot' not in features.columns)
        self.assertTrue('time_until_disrupt' not in features.columns)


class TestLoadFeatureList(unittest.TestCase):
    """Tests for the function load_feature_list()"""

    def test_no_duplicate_features(self):
        """Ensure that there are no duplicate features in the feature list
        """

        feature_list = load_feature_list(TEST_DEVICE, TEST_DATASET_PATH)
        self.assertEqual(len(feature_list), len(set(feature_list)))

    def test_all_features_covered(self):
        """Ensure that all features are covered by the feature list
        """

        feature_list = load_feature_list(TEST_DEVICE, TEST_DATASET_PATH)

        # Ensure that all features are covered
        for feature in ['ip', 'n_e', 'aminor', 'kappa', 'squareness', 'delta', 'li', 'Wmhd']:
            self.assertTrue(feature in feature_list)

    def test_no_non_features(self):
        """Ensure 'shot', 'time', and 'time_until_disrupt' are not returned as features"""

        feature_list = load_feature_list(TEST_DEVICE, TEST_DATASET_PATH)

        self.assertFalse('shot' in feature_list)
        self.assertFalse('time' in feature_list)
        self.assertFalse('time_until_disrupt' in feature_list)
        
class TestShotLists(unittest.TestCase):
    """Tests for the functions load_shot_list(), load_disruptive_shot_list(), and load_non_disruptive_shot_list()"""

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
