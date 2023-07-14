"""File for plotting the results of the experiments."""

import numpy as np
import matplotlib.pyplot as plt

from Experiments import Experiment


# Plots for 

def plot_roc_auc_vs_horizon_macro(experiment_list:list[Experiment], horizons=np.linspace(0.01, 0.4, 9)):
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
        

def plot_roc_auc_vs_horizon_micro(experiment_list:list[Experiment], horizons=np.linspace(0.01, 0.4, 10)):
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

def plot_warning_time_vs_FPR():
    """ Averaged over all shots

    """

def plot_warning_time_vs_precision():
    """ Averaged over all shots

    """


# Plots for comparing output of models to time series of individual shots

def plot_risk():
    """ Compared with Ip or actual disruption time or something """

def plot_expected_lifetime():
    """ NOT RIGOROUS """

