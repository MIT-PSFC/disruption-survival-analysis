import unittest

from tests.test_manage_datasets import TEST_DEVICE, TEST_DATASET_PATH

from disruption_survival_analysis.plot_experiments import *
from disruption_survival_analysis.experiment_utils import load_experiment_config
from disruption_survival_analysis.Experiments import Experiment

TEST_MODEL_TYPES = ['rf']
TEST_ALARM_TYPES = ['sthr']
TEST_METRICS = ['auroc']
TEST_MIN_WARNING_TIMES = [0.02]


class TestPlotExperiments(unittest.TestCase):
    """Tests for `plot_experiments.py`.
    Ensures that all plots can be run without throwing errors.
    """

    def setUp(self):
        # Create a list of experiments
        self.experiments = []
        for model in TEST_MODEL_TYPES:
            for alarm in TEST_ALARM_TYPES:
                for metric in TEST_METRICS:
                    for min_warning_time in TEST_MIN_WARNING_TIMES:
                        
                        config = load_experiment_config(TEST_DEVICE, TEST_DATASET_PATH, model, alarm, metric, min_warning_time)
                        
                        # Create test experiment from config
                        experiment = Experiment(config, 'test')
                        self.experiments.append(experiment)

    def test_plot_auroc_timeslice_all_vs_horizon(self):

        plot_auroc_timeslice_all_vs_horizon(self.experiments, disrupt_only=True, test=True)
        plot_auroc_timeslice_all_vs_horizon(self.experiments, disrupt_only=False, test=True)

    def test_plot_auroc_timeslice_shot_avg_vs_horizon(self):

        plot_auroc_timeslice_shot_avg_vs_horizon(self.experiments, test=True)

"""

    def test_plot_false_positive_vs_threshold(self):

        plot_false_positive_vs_threshold(self.experiments, cutoff_far=0.05, method='average')

"""