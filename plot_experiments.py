"""File for plotting the results of the experiments."""

import numpy as np
import matplotlib.pyplot as plt

from Experiments import Experiment

from manage_datasets import load_features_outcomes
from auton_survival.metrics import survival_regression_metric

DEFAULT_HORIZONS = np.linspace(0.01, 0.4, 5)
# TODO fix horizons
DEFAULT_HORIZONS[1] = 0.05
DEFAULT_HORIZONS[2] = 0.1
DEFAULT_HORIZONS[3] = 0.2
MAX_WARNING_TIME_MS = 1000

MINIMUM_WARNING_TIME = 0.02 # From Ryan, we need at least 20ms to react
GOOD_WARNING_TIME = 0.1 # Also from Ryan, would be very nice to have 100ms to react


def plot_brier_score_vs_horizon(experiment_list:list[Experiment], horizons=DEFAULT_HORIZONS):

    horizons_ms = horizons*1000

    plt.figure()
    
    for experiment in experiment_list:
        device = experiment.device
        dataset_path = experiment.dataset_path
        
        _, y_tr = load_features_outcomes(device, dataset_path, 'train', experiment.predictor.features)
        x_test, y_test = load_features_outcomes(device, dataset_path, 'test', experiment.predictor.features)

        survival_predictions = experiment.predictor.model.predict_survival(x_test, horizons)

        if experiment.predictor.model.model in ['cph']:
            brier_score = survival_regression_metric('brs', outcomes=y_test, predictions=survival_predictions, 
                                                    times=horizons, outcomes_train=y_tr)
        elif experiment.predictor.model.model in ['dsm']:
            brier_score = survival_regression_metric('ibs', outcomes=y_test, predictions=survival_predictions, 
                                                    times=horizons, outcomes_train=y_tr)
        else:
            brier_score = None
        plt.plot(horizons_ms, brier_score, label=experiment.name)
        #plt.fill_between(horizons, brier_score - brier_score_std, brier_score + brier_score_std, alpha=0.2)

    plt.xlim([horizons_ms[0], horizons_ms[-1]])
    plt.ylim([0, 1])

    plt.xticks(horizons_ms)

    plt.xlabel('Horizon [ms]')
    plt.ylabel('Brier Score')

    plt.title('Brier Score vs. Horizon')

    plt.legend()
    plt.show()

# Plots for model performance tests (in terms of ROC AUC, TAR, FAR, Warning Time, etc.)

def plot_roc_auc_vs_horizon_macro(experiment_list:list[Experiment], horizons=DEFAULT_HORIZONS):
    """ Averaged over all shots
    
    """

    horizons_ms = horizons*1000

    plt.figure()
    
    for experiment in experiment_list:
        roc_auc, roc_auc_std = experiment.roc_auc_macro(horizons)
        plt.errorbar(horizons_ms, roc_auc, yerr=roc_auc_std, label=experiment.name, fmt='o-', capsize=5)
        #plt.plot(horizons, roc_auc, label=experiment.name)
        #plt.fill_between(horizons, roc_auc - roc_auc_std, roc_auc + roc_auc_std, alpha=0.2)

    plt.xlim([horizons_ms[0], horizons_ms[-1]])
    plt.ylim([0.5, 1])

    plt.xticks(horizons_ms)

    plt.xlabel('Horizon [ms]')
    plt.ylabel('Macro Average Area Under ROC Curve')

    plt.title('Macro Average ROC AUC vs. Horizon')

    plt.legend()
    plt.show()

def plot_TAR_vs_FAR(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0], required_warning_time=MINIMUM_WARNING_TIME):
    """ Averaged over all shots
    
    """

    plt.figure()

    # This gets funky because I want to plot the TAR as 0, 0.9, 0.99, 0.999, 1.0

    plt.yscale('log')

    for experiment in experiment_list:
        false_alarm_rates, true_alarm_rates = experiment.true_alarm_rate_vs_false_alarm_rate(horizon, required_warning_time)
        plt.plot(false_alarm_rates, 1-true_alarm_rates, label=experiment.name)

    
    plt.gca().invert_yaxis()
    plt.gca().set_yticklabels(1-plt.gca().get_yticks())


    # Set x axis to be logarithmic scale (to better show the low false alarm rates)
    plt.xscale('log')

    plt.xlim([0, 1])
    plt.ylim([-5, 0])

    plt.xlabel('False Alarm Rate')
    plt.ylabel('True Alarm Rate')

    plt.title(f"{required_warning_time * 1000} ms True Alarm Rate vs. False Alarm Rate at {horizon*1000} ms Horizon")

    plt.legend()
    plt.show()
        

def plot_roc_auc_vs_horizon_micro(experiment_list:list[Experiment], horizons=DEFAULT_HORIZONS):
    """ Averaged over a single shot

    """

    horizons_ms = horizons*1000

    plt.figure()

    for experiment in experiment_list:
        roc_auc = experiment.roc_auc_micro_all(horizons)
        plt.plot(horizons_ms, roc_auc, label=experiment.name)

    plt.xlim([horizons_ms[0], horizons_ms[-1]])
    plt.ylim([0.5, 1])

    plt.xticks(horizons_ms)

    plt.xlabel('Horizon [ms]')
    plt.ylabel('Micro Average Area Under ROC Curve')

    plt.title('Micro Average ROC AUC vs. Horizon')

    plt.legend()
    plt.show()

def plot_missed_alarm_rate_vs_threshold(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0]):
    """ Averaged over all shots
    
    """

    plt.figure()

    for experiment in experiment_list:
        thresholds, true_alarm_rates = experiment.true_alarm_rate_vs_threshold(horizon)
        missed_alarm_rates = 1 - true_alarm_rates
        # Plot false alarm rate vs threshold, where threshold is on a log scale
        plt.loglog(thresholds, missed_alarm_rates, label=experiment.name)

    plt.xlim([min(thresholds), 1])
    plt.ylim([0, 1])

    plt.xlabel('Threshold')
    plt.ylabel('Missed Alarm Rate')

    plt.title(f'Missed Alarm Rate vs. Threshold at {horizon*1000:.0f} ms Horizon')

    plt.legend()
    plt.show()

def plot_FAR_vs_threshold(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0]):
    """ Averaged over all shots
    
    """

    plt.figure()

    for experiment in experiment_list:
        thresholds, false_alarm_rates = experiment.false_alarm_rate_vs_threshold(horizon)
        # Plot false alarm rate vs threshold, where threshold is on a log scale
        plt.semilogx(thresholds, false_alarm_rates, label=experiment.name)

    plt.xlim([min(thresholds), 1])
    plt.ylim([0, 1])

    plt.xlabel('Threshold')
    plt.ylabel('False Alarm Rate')

    plt.title(f'False Alarm Rate vs. Threshold at {horizon*1000:.0f} ms Horizon')

    plt.legend()
    plt.show()

def plot_missed_alarm_rate_vs_false_alarm_rate(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0]):
    """ Averaged over all shots
    
    """

    plt.figure()

    for experiment in experiment_list:
        false_alarm_rates, missed_alarm_rates = experiment.missed_alarm_rate_vs_false_alarm_rate(horizon)
        plt.semilogy(false_alarm_rates, missed_alarm_rates, label=experiment.name)

    plt.xlim([0, 1])
    plt.ylim([1e-3, 1])

    plt.xlabel('False Alarm Rate')
    plt.ylabel('Missed Alarm Rate')

    plt.title(f'Missed Alarm Rate vs. False Alarm Rate at {horizon*1000:.0f} ms Horizon')

    plt.legend()
    plt.show()


# Warning time plots (maybe not the greatest metric due to the threshold sweeping from 0)

def plot_warning_time_vs_threshold(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0]):
    """ Averaged over all shots
    """

    horizon_ms = horizon*1000

    plt.figure()

    for experiment in experiment_list:
        thresholds, warning_time_avg, warning_time_std = experiment.warning_time_vs_threshold(horizon)
        warning_time_avg_ms = [i * 1000 for i in warning_time_avg]
        warning_time_std_ms = [i * 1000 for i in warning_time_std]
        # TODO: reintroduce error bars
        plt.semilogx(thresholds, warning_time_avg_ms, label=experiment.name)

    plt.xlim([min(thresholds), max(thresholds)])
    plt.ylim([0, MAX_WARNING_TIME_MS])

    plt.xlabel('Threshold')
    plt.ylabel('Warning Time [ms]')

    plt.title(f'Warning Time vs. Threshold at {horizon_ms:.0f} ms Horizon')

    plt.legend()
    plt.show()

def plot_warning_time_vs_TAR(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0]):
    """ Averaged over all shots
    """

    horizon_ms = horizon*1000

    plt.figure()

    for experiment in experiment_list:
        true_alarm_rates, warning_time_avg, warning_time_std = experiment.warning_time_vs_true_alarm_rate(horizon)
        warning_time_avg_ms = [i * 1000 for i in warning_time_avg]
        warning_time_std_ms = [i * 1000 for i in warning_time_std]
        # TODO: reintroduce error bars
        plt.plot(true_alarm_rates, warning_time_avg_ms, label=experiment.name)

    plt.xlim([min(true_alarm_rates), max(true_alarm_rates)])
    plt.ylim([0, MAX_WARNING_TIME_MS])

    plt.xlabel('True Alarm Rate')
    plt.ylabel('Warning Time [ms]')

    plt.title(f'Warning Time vs. True Alarm Rate at {horizon_ms:.0f} ms Horizon')

    plt.legend()
    plt.show()

def plot_warning_time_vs_FAR(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0], min_far=None, max_far=None, min_warning_time=None, max_warning_time=None):
    """ Averaged over all shots
    """

    horizon_ms = horizon*1000

    plt.figure()

    for experiment in experiment_list:
        false_alarm_rates, warning_time_avg, warning_time_std = experiment.warning_time_vs_false_alarm_rate(horizon)
        warning_time_avg_ms = [i * 1000 for i in warning_time_avg]
        warning_time_std_ms = [i * 1000 for i in warning_time_std]
        # TODO: reintroduce error bars
        plt.semilogx(false_alarm_rates, warning_time_avg_ms, label=experiment.name)

    # TODO: changed temporarily

    if min_far is None:
        min_far = min(false_alarm_rates)
    if max_far is None:
        max_far = max(false_alarm_rates)
    if min_warning_time is None:
        min_warning_time = 0
    if max_warning_time is None:
        max_warning_time = MAX_WARNING_TIME_MS

    plt.xlim([min_far, max_far])
    plt.ylim([min_warning_time, max_warning_time])

    plt.xlabel('False Alarm Rate')
    plt.ylabel('Warning Time [ms]')

    plt.title(f'Warning Time vs. False Alarm Rate at {horizon_ms:.0f} ms Horizon')

    plt.legend()
    plt.show()

def plot_warning_time_vs_missed_alarm_rate(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0], min_missed_alarm_rate=None, max_missed_alarm_rate=None, min_warning_time=None, max_warning_time=None):
    
    horizon_ms = horizon*1000

    plt.figure()

    for experiment in experiment_list:
        missed_alarm_rates, warning_time_avg, warning_time_std = experiment.warning_time_vs_missed_alarm_rate(horizon)
        warning_time_avg_ms = [i * 1000 for i in warning_time_avg]
        warning_time_std_ms = [i * 1000 for i in warning_time_std]
        plt.semilogx(missed_alarm_rates, warning_time_avg_ms, label=experiment.name)

    if min_missed_alarm_rate is None:
        min_missed_alarm_rate = min(missed_alarm_rates)
    if max_missed_alarm_rate is None:
        max_missed_alarm_rate = max(missed_alarm_rates)
    if min_warning_time is None:
        min_warning_time = 0
    if max_warning_time is None:
        max_warning_time = MAX_WARNING_TIME_MS

    plt.xlim([min_missed_alarm_rate, max_missed_alarm_rate])
    plt.ylim([min_warning_time, max_warning_time])

    plt.xlabel('Missed Alarm Rate')
    plt.ylabel('Warning Time [ms]')

    plt.title(f'Warning Time vs. Missed Alarm Rate at {horizon_ms:.0f} ms Horizon')

    plt.legend()    

def plot_warning_time_vs_precision(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0], scaled=True):
    """ Averaged over all shots

    """

    horizon_ms = horizon*1000

    plt.figure()

    for experiment in experiment_list:
        precision, warning_time_avg, warning_time_std = experiment.warning_vs_precision(horizon, scaled=scaled)
        if not scaled:
            warning_time_avg_ms = [i * 1000 for i in warning_time_avg]
            warning_time_std_ms = [i * 1000 for i in warning_time_std]
            # TODO: reintroduce error bars
            plt.plot(warning_time_avg_ms, precision, label=experiment.name)
        else:
            plt.plot(warning_time_avg, precision, label=experiment.name)

    plt.ylim([min(precision), max(precision)])
    if not scaled:
        plt.xlim([0, MAX_WARNING_TIME_MS])
    else:
        plt.xlim([0, 1])

    plt.ylabel('Precision (True Alarms/(Total Alarms))')
    if scaled:
        plt.xlabel('Warning Time Fraction of Shot Duration')
    else:
        plt.xlabel('Warning Time [ms]')

    plt.title(f'Precision vs. Warning Time at {horizon_ms:.0f} ms Horizon')

    plt.legend()
    plt.show()

# Plots for comparing output of models to time series of individual shots

def plot_risk_compare_horizons(experiment:Experiment, shot, horizons=DEFAULT_HORIZONS):
    """ Compared with Ip or actual disruption time or something """

    horizons_ms = horizons*1000

    plt.figure()

    # Invert horizons so the legend matches the order of the lines
    horizons = horizons[::-1]

    for horizon in horizons:
        time = experiment.get_time(shot)
        risk = experiment.get_risk(shot, horizon)
        plt.plot(time, risk, label=f'{(horizon*1000):.0f} ms')

    plt.xlim([time[0], time[-1]])
    plt.ylim([0, 1])

    plt.xlabel('Time [ms]')
    plt.ylabel('Disruption Risk')

    plt.title(f'{experiment.name} Disruption Risk vs. Time at various Horizons for Shot {shot}')

    plt.legend()

def plot_risk_compare_models(experiment_list:list[Experiment], shot, horizon):
    pass

def plot_expected_lifetime():
    """ NOT RIGOROUS """
    pass

# Plots for showing the composition of the dataset

def plot_all_shot_durations(experiment:Experiment):

    shot_durations_ms = experiment.get_all_shot_durations()*1000

    plt.figure()

    plt.hist(shot_durations_ms, bins=50)

    plt.xlabel('Shot Duration [ms]')
    plt.ylabel('Number of Shots')

    plt.title(f'Shot Durations for {experiment.dataset_path}')

    plt.show()

def plot_disruptive_vs_non_disruptive_shot_durations(experiment:Experiment):

    disruptive_shot_durations_ms = experiment.get_disruptive_shot_durations()*1000
    non_disruptive_shot_durations_ms = experiment.get_non_disruptive_shot_durations()*1000

    plt.figure()

    plt.hist(disruptive_shot_durations_ms, bins=50, label='Disruptive')
    plt.hist(non_disruptive_shot_durations_ms, bins=50, label='Non-Disruptive')

    plt.xlim([500, max(max(disruptive_shot_durations_ms), max(non_disruptive_shot_durations_ms))])

    plt.xlabel('Shot Duration [ms]')
    plt.ylabel('Number of Shots')
    plt.title(f'Shot Durations for {experiment.dataset_path}')
    plt.legend()
    plt.show()

"""

This plot does not make sense because there are many warning times for each shot
(one for each threshold)

def plot_shot_duration_vs_warning_time(experiment:Experiment, horizon):

    disruptive_shot_duration_ms = experiment.get_disruptive_shot_durations()*1000
    warning_time_ms = experiment.get_warning_times_list(horizon)*1000

    unique_warning_time, avg_shot_duration, _ = clump_many_to_one_statistics(warning_time_ms, disruptive_shot_duration_ms)

    plt.figure()

    plt.plot(unique_warning_time, avg_shot_duration)

    plt.xlim([0, max(unique_warning_time)])
    plt.ylim([0, max(avg_shot_duration)])
    

    plt.xlabel('Warning Time [ms]')
    plt.ylabel('Avg Shot Duration [ms]')
    plt.title(f'Shot Duration vs Warning Time for {experiment.dataset_path} at {horizon*1000:.0f} ms Horizon')

    plt.show()
"""
