"""File for plotting the results of the experiments."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from disruption_survival_analysis.Experiments import Experiment

LEGEND_FONT_SIZE = 12
TICK_FONT_SIZE = 18
LABEL_FONT_SIZE = 22
TITLE_FONT_SIZE = 24

#PLOT_STYLE = 'seaborn-v0_8-colorblind'
PLOT_STYLE = 'seaborn-v0_8-poster'

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

    # Put x on symlog, where the x axis under 0.01 is linear
    plt.xscale('symlog', linthresh=0.01)

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

    # Put x on symlog, where the x axis under 0.01 is linear
    plt.xscale('symlog', linthresh=0.01)

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
    #  Put x on symlog, where the x axis under 0.01 is linear
    plt.xscale('symlog', linthresh=0.01)

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

    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    plt.grid(True, color='w', linestyle='-', linewidth=1.5)
    plt.gca().patch.set_facecolor('0.92')

    for experiment in experiment_list:
        false_alarm_rates, true_alarm_rates = experiment.true_alarm_rates_vs_false_alarm_rates()
        plt.plot(false_alarm_rates, true_alarm_rates, label=pretty_name(experiment.name), linewidth=2)

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel("False Alarm Rate", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("True Alarm Rate", fontsize=LABEL_FONT_SIZE)

    # Set x tick font size
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)

    # Put ticks at percents
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1],
                ["0\%", "20\%", "40\%", "60\%", "80\%", "100\%"])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1],
                ["0\%", "20\%", "40\%", "60\%", "80\%", "100\%"])

    plt.title("C-Mod, 100ms Required Warning Time", fontsize=TITLE_FONT_SIZE)

    if not test:
        plt.show()

    plt.rcParams.update(mpl.rcParamsDefault)

def plot_missed_alarm_rates_vs_false_alarm_rates(experiment_list:list[Experiment], test=False):
    """ Plot the log of missed alarm rate vs false alarm rates for each experiment in the list.
    
    """

    plt.figure()

    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.grid(True, color='w', linestyle='-', linewidth=1.5)
    plt.gca().patch.set_facecolor('0.92')

    # Check what type of plot this is going to be (comparing models or comparing required warning times)
    if len(experiment_list) == 1:
        plot_type = 'no_title'
    else:
        first_time = experiment_list[0].predictor.trained_required_warning_time
        second_time = experiment_list[1].predictor.trained_required_warning_time
        if  first_time != second_time:
            plot_type = 'comparing_required_warning_times'
        else:
            plot_type = 'comparing_models'
            plt.title(f"Required Warning Time: {first_time*1000:.0f} ms", fontsize=TITLE_FONT_SIZE)


    for experiment in experiment_list:
        false_alarm_rates, missed_alarm_rates = experiment.missed_alarm_rates_vs_false_alarm_rates()
        if plot_type == 'comparing_models':
            name = pretty_name(experiment.name)
        elif plot_type == 'comparing_required_warning_times':
            name = pretty_name(experiment.name, required_warning_times=True)
        plt.plot(false_alarm_rates, missed_alarm_rates, label=name, linewidth=2)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.yscale('symlog', linthresh=0.1)

    # Put ticks at 1%, 5%, 10%
    plt.yticks([0.01, .05, .1, .5],
               ["1\%", "5\%", "10\%", "50\%"])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1],
                ["0\%", "20\%", "40\%", "60\%", "80\%", "100\%"])
    
    # Make ticks bigger
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)

    #plt.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
    plt.xlabel("False Alarm Rate", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Missed Alarm Rate", fontsize=LABEL_FONT_SIZE)

    if not test:
        plt.show()

    plt.rcParams.update(mpl.rcParamsDefault)


def plot_avg_warning_times_vs_false_alarm_rates(experiment_list:list[Experiment], required_warning_time, test=False):
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

    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.grid(True, color='w', linestyle='-', linewidth=1.5)
    plt.gca().patch.set_facecolor('0.92')

    for experiment in experiment_list:
        false_alarm_rates, avg_warning_times, std_warning_times = experiment.warning_times_vs_false_alarm_rates()
        plt.plot(false_alarm_rates, avg_warning_times*1000, label=experiment.name, linewidth=2)

    plt.xlim([0, 0.05])
    plt.ylim([0, 120])

    plt.xscale('symlog')

    # Make x ticks go from 0 to 0.05 in increments of 0.01
    plt.xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05],
                ["0\%", "1\%", "2\%", "3\%", "4\%", "5\%"])
    
    # Make ticks bigger
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)

    #plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel("False Alarm Rate", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Average Warning Time [ms]", fontsize=LABEL_FONT_SIZE)

    # Put a vertical line at 0.05 false alarm rate
    #plt.axvline(x=0.05, color='r', linestyle='--')

    # Put a horizontal line at the required warning time
    #plt.axhline(y=required_warning_time*1000, color='k', linestyle='--')

    if not test:
        plt.show()

    plt.rcParams.update(mpl.rcParamsDefault)

# Expected Time To Disruption

def plot_expected_time_to_disruption_shot(experiment_list:list[Experiment], shot_number, test=False):
    """Plot the expected time to disruption for a given shot for each experiment in the list."""

    plt.figure()

    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    times = experiment_list[0].get_shot_data(shot_number)['time'].values

    # Set boolean disruptive if shot number is in the disruptive shot list
    disruptive = shot_number in experiment_list[0].get_disruptive_shot_list()

    for experiment in experiment_list:
        ettd = experiment.get_expected_time_to_disruption_shot(shot_number)
        plt.plot(times, ettd, label=pretty_name(experiment.name))

    plt.xlim([min(times), max(times)])
    plt.ylim([0, 6.1])

    # Set tick font sizes
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)

    #plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel("Time [s]", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Expected Time To Disruption [s]", fontsize=LABEL_FONT_SIZE)

    if disruptive:
        plt.title(f"Shot {int(shot_number)}, Disrupted", fontsize=TITLE_FONT_SIZE)
    else:
        plt.title(f"Shot {int(shot_number)}, Not Disrupted", fontsize=TITLE_FONT_SIZE)

    if not test:
        plt.show()

    plt.rcParams.update(mpl.rcParamsDefault)



def pretty_name(experiment_name:str, alarm_types=False, metrics=False, required_warning_times=False):
    """ Return a pretty name for the experiment name."""

    split_sections = experiment_name.split("_")

    pretty_name = ""
    # Set the first part based on the model type
    if split_sections[0] == "rf":
        pretty_name += "Random Forest"
    elif split_sections[0] == "km":
        pretty_name += "Kaplan-Meier"
    elif split_sections[0] == "cph":
        pretty_name += "Cox Proportional Hazards"
    elif split_sections[0] == "dcph":
        pretty_name += "Deep Cox Proportional Hazards"
    elif split_sections[0] == "dsm":
        pretty_name += "Deep Survival Machines"

    if required_warning_times:
        pretty_name += " " + split_sections[-1]

    return pretty_name