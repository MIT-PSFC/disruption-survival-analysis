"""File for plotting the results of the experiments."""

import numpy as np
import matplotlib.pyplot as plt

from Experiments import Experiment


DEFAULT_HORIZONS = np.linspace(0.01, 0.4, 10)

# Plots for model performance tests (in terms of ROC AUC, TPR, FPR, Warning Time, etc.)

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

def plot_TPR_vs_threshold_macro():
    """ Averaged over all shots
    
    """

def plot_TPR_vs_threshold_micro():
    """ Averaged over a single shot

    """

def plot_TPR_vs_FPR_macro():
    """ Averaged over all shots
    
    """

def plot_TPR_vs_FPR_micro():
    """ Averaged over a single shot

    """

def plot_warning_time_vs_FPR(experiment_list:list[Experiment], horizon=DEFAULT_HORIZONS[0]):
    """ Averaged over all shots
    """

    horizon_ms = horizon*1000

    plt.figure()

    for experiment in experiment_list:
        fpr, warning_time_avg, warning_time_std = experiment.warning_vs_fpr(horizon)
        warning_time_avg_ms = [i * 1000 for i in warning_time_avg]
        warning_time_std_ms = [i * 1000 for i in warning_time_std]
        plt.errorbar(fpr, warning_time_avg_ms, yerr=warning_time_std_ms, label=experiment.name, fmt='o-')

    plt.xlim([0, 1])
    plt.ylim([0, max(warning_time_avg_ms) + max(warning_time_std_ms)])

    plt.xlabel('False Positive Rate')
    plt.ylabel('Warning Time [ms]')

    plt.title(f'Warning Time vs. False Positive Rate at {horizon_ms:.0f} ms Horizon')

    plt.legend()
    plt.show()

def plot_warning_time_vs_precision():
    """ Averaged over all shots

    """


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
