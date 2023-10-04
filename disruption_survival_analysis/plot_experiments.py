"""File for plotting the results of the experiments."""

import matplotlib.pyplot as plt

from disruption_survival_analysis.Experiments import Experiment

LEGEND_FONT_SIZE = 14
TICK_FONT_SIZE = 14
LABEL_FONT_SIZE = 16
TITLE_FONT_SIZE = 18

# Plots of the metrics vs thresholds

def plot_true_alarm_rates_vs_thresholds(experiment_list:list[Experiment], test=False):
    """ Plot the true alarm rate vs threshold for each experiment in the list.

    Parameters
    ----------
    experiment_list : list of Experiment
        The list of experiments to plot
    test : bool, optional
        If True, will run all the calculations but won't display the plot.
        Allows all tests to run uninterrupted.
    """

    plt.figure()

    for experiment in experiment_list:
        thresholds, false_alarm_rates = experiment.true_alarm_rates_vs_thresholds()
        plt.plot(thresholds, false_alarm_rates, label=experiment.name)

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel("Threshold", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("True Alarm Rate", fontsize=LABEL_FONT_SIZE)

    if not test:
        plt.show()

def plot_false_alarm_rates_vs_thresholds(experiment_list:list[Experiment], test=False):
    """ Plot the false alarm rate vs threshold for each experiment in the list.

    Parameters
    ----------
    experiment_list : list of Experiment
        The list of experiments to plot
    test : bool, optional
        If True, will run all the calculations but won't display the plot.
        Allows all tests to run uninterrupted.
    """

    plt.figure()

    for experiment in experiment_list:
        thresholds, false_alarm_rates = experiment.false_alarm_rates_vs_thresholds()
        plt.plot(thresholds, false_alarm_rates, label=experiment.name)

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel("Threshold", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("False Alarm Rate", fontsize=LABEL_FONT_SIZE)

    if not test:
        plt.show()

def plot_warning_times_vs_thresholds(experiment_list:list[Experiment], test=False):
    """ Plot the warning time vs threshold for each experiment in the list.

    Parameters
    ----------
    experiment_list : list of Experiment
        The list of experiments to plot
    test : bool, optional
        If True, will run all the calculations but won't display the plot.
        Allows all tests to run uninterrupted.
    """

    plt.figure()

    for experiment in experiment_list:
        thresholds, avg_warning_times, std_warning_times = experiment.warning_times_vs_thresholds()
        plt.plot(thresholds, avg_warning_times, label=experiment.name)

    plt.xlim([0, 1])

    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel("Threshold", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Average Warning Time", fontsize=LABEL_FONT_SIZE)

    if not test:
        plt.show()

# Plots of metrics vs false alarm rates

def plot_true_alarm_rates_vs_false_alarm_rates(experiment_list:list[Experiment], test=False):
    """ Plot the true alarm rate vs false alarm rate for each experiment in the list.

    Parameters
    ----------
    experiment_list : list of Experiment
        The list of experiments to plot
    test : bool, optional
        If True, will run all the calculations but won't display the plot.
        Allows all tests to run uninterrupted.
    """

    plt.figure()

    for experiment in experiment_list:
        false_alarm_rates, true_alarm_rates = experiment.true_alarm_rates_vs_false_alarm_rates()
        plt.plot(false_alarm_rates, true_alarm_rates, label=experiment.name)

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel("False Alarm Rate", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("True Alarm Rate", fontsize=LABEL_FONT_SIZE)

    if not test:
        plt.show()

def plot_avg_warning_times_vs_false_alarm_rates(experiment_list:list[Experiment], test=False):
    """ Plot the average warning time vs false alarm rate for each experiment in the list.

    Parameters
    ----------
    experiment_list : list of Experiment
        The list of experiments to plot
    test : bool, optional
        If True, will run all the calculations but won't display the plot.
        Allows all tests to run uninterrupted.
    """

    plt.figure()

    for experiment in experiment_list:
        false_alarm_rates, avg_warning_times, std_warning_times = experiment.warning_times_vs_false_alarm_rates()
        plt.plot(false_alarm_rates, avg_warning_times, label=experiment.name)

    plt.xlim([0, 1])

    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel("False Alarm Rate", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Average Warning Time", fontsize=LABEL_FONT_SIZE)

    if not test:
        plt.show()