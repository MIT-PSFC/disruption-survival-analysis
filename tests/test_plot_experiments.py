import unittest

from disruption_survival_analysis.plot_experiments import *
from disruption_survival_analysis.experiment_utils import load_experiment_config
from disruption_survival_analysis.Experiments import Experiment

#TEST_DEVICE = 'synthetic'
#TEST_DATASET = 'synthetic100'

TEST_DEVICE = 'cmod'
TEST_DATASET = 'preliminary_dataset_no_ufo'

TEST_MODEL_TYPES = ['dsm']
TEST_ALARM_TYPES = ['sthr']
TEST_METRICS = ['auroc']
TEST_MIN_WARNING_TIMES = [0.1]


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
                        
                        config = load_experiment_config(TEST_DEVICE, TEST_DATASET, model, alarm, metric, min_warning_time)
                        
                        # Create test experiment from config
                        experiment = Experiment(config, 'test')
                        self.experiments.append(experiment)

    def test_plot_auroc_timeslice_all_vs_horizon(self):

        plot_auroc_timeslice_all_vs_horizon(self.experiments, disrupt_only=True)
        plot_auroc_timeslice_all_vs_horizon(self.experiments, disrupt_only=False)

    def test_plot_auroc_timeslice_shot_avg_vs_horizon(self):

        plot_auroc_timeslice_shot_avg_vs_horizon(self.experiments)


