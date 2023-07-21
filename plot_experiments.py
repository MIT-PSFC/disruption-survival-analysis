"""File for plotting the results of the experiments."""

import numpy as np
import matplotlib.pyplot as plt

from Experiments import Experiment

DEFAULT_HORIZONS = np.linspace(0.001, 0.4, 6)
# TODO fix horizons
DEFAULT_HORIZONS[1] = 0.01
DEFAULT_HORIZONS[2] = 0.05
DEFAULT_HORIZONS[3] = 0.1
DEFAULT_HORIZONS[4] = 0.2
MAX_WARNING_TIME_MS = 1000

# Plots for model performance tests (in terms of ROC AUC, TAR, FAR, Warning Time, etc.)

def plot_roc_auc_vs_horizon_macro(experiment_list:list[Experiment], horizons=DEFAULT_HORIZONS):
    """ Averaged over all shots
    
    """

    horizons_ms = horizons*1000

    plt.figure()
    
    for experiment in experiment_list:
        roc_auc, roc_auc_std = experiment.roc_auc_macro(horizons)
        plt.errorbar(horizons_ms, roc_auc, yerr=roc_auc_std, label=experiment.name, fmt='o-')
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

def plot_TAR_vs_FAR(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0]):
    """ Averaged over all shots
    
    """

    plt.figure()

    for experiment in experiment_list:
        false_alarm_rates, true_alarm_rates = experiment.true_alarm_rate_vs_false_alarm_rate(horizon)
        plt.plot(false_alarm_rates, true_alarm_rates, label=experiment.name)

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.xlabel('False Alarm Rate')
    plt.ylabel('True Alarm Rate')

    plt.title(f"True Alarm Rate vs. False Alarm Rate at {horizon*1000} ms Horizon")

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

def plot_TAR_vs_threshold(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0]):
    """ Averaged over all shots
    
    """

    plt.figure()

    for experiment in experiment_list:
        thresholds, true_alarm_rates = experiment.true_alarm_rate_vs_threshold(horizon)
        # Plot false alarm rate vs threshold, where threshold is on a log scale
        plt.semilogx(thresholds, true_alarm_rates, label=experiment.name)

    plt.xlim([min(thresholds), 1])
    plt.ylim([0, 1])

    plt.xlabel('Threshold')
    plt.ylabel('True Alarm Rate')

    plt.title(f'True Alarm Rate vs. Threshold at {horizon*1000:.0f} ms Horizon')

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

def plot_warning_time_vs_FAR(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0]):
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

    plt.xlim([min(false_alarm_rates), max(false_alarm_rates)])
    plt.ylim([0, MAX_WARNING_TIME_MS])

    plt.xlabel('False Alarm Rate')
    plt.ylabel('Warning Time [ms]')

    plt.title(f'Warning Time vs. False Alarm Rate at {horizon_ms:.0f} ms Horizon')

    plt.legend()
    plt.show()

def plot_warning_time_vs_precision(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0]):
    """ Averaged over all shots

    """

    horizon_ms = horizon*1000

    plt.figure()

    for experiment in experiment_list:
        precision, warning_time_avg, warning_time_std = experiment.warning_vs_precision(horizon)
        warning_time_avg_ms = [i * 1000 for i in warning_time_avg]
        warning_time_std_ms = [i * 1000 for i in warning_time_std]
        # TODO: reintroduce error bars
        plt.plot(precision, warning_time_avg_ms, label=experiment.name)

    plt.xlim([min(precision), max(precision)])
    plt.ylim([0, MAX_WARNING_TIME_MS])

    plt.xlabel('Precision (True Alarms/(Total Alarms))')
    plt.ylabel('Warning Time [ms]')

    plt.title(f'Warning Time vs. Precision at {horizon_ms:.0f} ms Horizon')

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
