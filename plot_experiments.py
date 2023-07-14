"""File for plotting the results of the experiments."""

import numpy as np
import matplotlib.pyplot as plt

from Experiments import Experiment


# Plots for 

def plot_roc_auc_vs_horizon_macro(experiment_list:list[Experiment], horizons=np.linspace(0, 1, 100)):
    """ Averaged over all shots
    
    """

    plt.figure()
    
    for experiment in experiment_list:
        roc_auc, roc_auc_std = experiment.roc_auc_macro(horizons)
        plt.errorbar(horizons, roc_auc, yerr=roc_auc_std, label=experiment.name, fmt='o-')
        #plt.plot(horizons, roc_auc, label=experiment.name)
        #plt.fill_between(horizons, roc_auc - roc_auc_std, roc_auc + roc_auc_std, alpha=0.2)

    plt.xlabel('Horizon [s]')
    plt.ylabel('Macro Average Area Under ROC Curve')

    plt.title('Macro Average ROC AUC vs. Horizon')

    plt.legend()
    plt.show()
        

def plot_roc_auc_vs_horizon_micro():
    """ Averaged over a single shot

    """

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

