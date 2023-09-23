"""File for plotting the results of the experiments."""

import numpy as np
import matplotlib.pyplot as plt

from disruption_survival_analysis.Experiments import Experiment

from disruption_survival_analysis.experiment_utils import make_shot_lifetime_curve

DEFAULT_HORIZONS = np.linspace(0.01, 0.4, 5)
# TODO fix horizons
DEFAULT_HORIZONS[1] = 0.05
DEFAULT_HORIZONS[2] = 0.1
DEFAULT_HORIZONS[3] = 0.2

MINIMUM_WARNING_TIME = 0.02 # From Ryan, we need at least 20ms to react
GOOD_WARNING_TIME = 0.1 # Also from Ryan, would be very nice to have 100ms to react

# Plots for timeslice-level model performance

def plot_auroc_timeslice_all_vs_horizon(experiment_list:list[Experiment], horizons=DEFAULT_HORIZONS, disrupt_only=True):
    """ Plot the full-dataset timeslice-level Area Under ROC Curve vs. horizon for each experiment.

    Parameters:
    ----------
    experiment_list: list[Experiment]
        List of experiments to plot
    horizons: list[float]
        List of horizons to evaluate the experiments at
    disrupt_only: bool
        Whether to only include disruptive shots in the evaluation
    """

    horizons_ms = horizons*1000

    plt.figure()

    for experiment in experiment_list:
        auroc = experiment.auroc_timeslice_all(horizons, disrupt_only)
        plt.plot(horizons_ms, auroc, label=experiment.name)

    plt.xlim([horizons_ms[0], horizons_ms[-1]])
    plt.ylim([0.5, 1])

    plt.xticks(horizons_ms)

    plt.xlabel('Horizon [ms]')
    plt.ylabel('Timeslice Area Under ROC Curve')

    if disrupt_only:
        plt.title('Disruptive-Only Timeslice Area Under ROC Curve vs. Horizon')
    else:
        plt.title('All shot Timeslice Area Under ROC Curve vs. Horizon')

    plt.legend()
    plt.show()

def plot_auroc_timeslice_shot_avg_vs_horizon(experiment_list:list[Experiment], horizons=DEFAULT_HORIZONS):
    """ Plot the shot-averaged timeslice Area Under ROC Curve vs. horizon for each experiment.
    Only includes disruptive shots, because a micro average over a non-disruptive shot cannot be done (only one class in truth value).

    Parameters:
    ----------
    experiment_list: list[Experiment]
        List of experiments to plot
    horizons: list[float]
        List of horizons to evaluate the experiments at
    """

    horizons_ms = horizons*1000

    plt.figure()
    
    for experiment in experiment_list:
        auroc_avg, auroc_std = experiment.auroc_timeslice_shot_avg(horizons)
        plt.errorbar(horizons_ms, auroc_avg, yerr=auroc_std, label=experiment.name, fmt='o-', capsize=5)

    plt.xlim([horizons_ms[0], horizons_ms[-1]])
    plt.ylim([0.5, 1])

    plt.xticks(horizons_ms)

    plt.xlabel('Horizon [ms]')
    plt.ylabel('Shot-Average Timeslice Area Under ROC Curve')

    plt.title('Shot-Average Timeslice Area Under ROC Curve vs. Horizon')

    plt.legend()
    plt.show()

# Plots for shot-level model performance

def plot_roc_curve(experiment_list:list[Experiment], horizon=None, required_warning_time=MINIMUM_WARNING_TIME):
    """ Averaged over all shots
    
    """

    plt.figure()

    for experiment in experiment_list:
        false_alarm_rates, true_alarm_rates = experiment.true_alarm_rate_vs_false_alarm_rate(horizon, required_warning_time)
        plt.plot(false_alarm_rates, true_alarm_rates, label=experiment.name)



    # Set x axis to be logarithmic scale (to better show the low false alarm rates)
    #plt.xscale('log')

    plt.xlim([0, 1])
    #plt.ylim([-5, 0])
    plt.ylim([0, 1])

    plt.xlabel('False Alarm Rate')
    plt.ylabel('True Alarm Rate')

    if horizon is None:
        plt.title(f"{required_warning_time * 1000} ms True Alarm Rate vs. False Alarm Rate")
    else:
        plt.title(f"{required_warning_time * 1000} ms True Alarm Rate vs. False Alarm Rate at {horizon*1000} ms Horizon")

    plt.legend()
    plt.show()

def plot_warning_time_vs_false_alarm_rate(experiment_list:list[Experiment], horizon=None, required_warning_time=MINIMUM_WARNING_TIME, min_far=None, max_far=None, min_warning_time=None, max_warning_time=None, cutoff_far=None, method='median'):
    """ Averaged over all shots
    """

    plt.figure()

    for experiment in experiment_list:
        false_alarm_rates, warning_time_avg, warning_time_std = experiment.warning_time_vs_false_alarm_rate(horizon, required_warning_time=None, method=method)
        warning_time_typical_ms = [i * 1000 for i in warning_time_avg]
        warning_time_spread_ms = [i * 1000 for i in warning_time_std]
        # TODO: reintroduce error bars
        # Plot with error bars
        plt.errorbar(false_alarm_rates, warning_time_typical_ms, yerr=warning_time_spread_ms, label=experiment.name, fmt='o-', capsize=5)

    # Make the x axis logarithmic
    plt.xscale('log')

    if min_far is None:
        min_far = min(false_alarm_rates)
    if max_far is None:
        max_far = max(false_alarm_rates)
    if min_warning_time is None:
        min_warning_time = 0
    if max_warning_time is None:
        max_warning_time = 500

    # Put a line at the required warning time
    plt.plot([min_far, max_far], [required_warning_time*1000, required_warning_time*1000], 'k--')

    plt.xlim([min_far, max_far])
    plt.ylim([min_warning_time, max_warning_time])

    if cutoff_far is not None:
        plt.axvline(x=cutoff_far, color='r', linestyle='--')

    plt.xlabel('False Alarm Rate')
    if method == 'median':
        plt.ylabel('Median Warning Time [ms]')
    elif method == 'average':
        plt.ylabel('Average Warning Time [ms]')

    plt.title(f'Warning Time vs. {int(required_warning_time*1000)}ms False Alarm Rate')

    plt.legend()
    plt.show()

def plot_warning_time_vs_threshold(experiment_list:list[Experiment], horizon=None, required_warning_time=MINIMUM_WARNING_TIME, min_threshold=None, max_threshold=None, min_warning_time=None, max_warning_time=None, cutoff_far=None, method='median'):
    """ Collected over all shots
    """

    plt.figure()

    for experiment in experiment_list:
        thresholds, warning_time_avg, warning_time_std = experiment.warning_time_vs_threshold(horizon, method=method)
        warning_time_avg_ms = [i * 1000 for i in warning_time_avg]
        warning_time_std_ms = [i * 1000 for i in warning_time_std]

        # Plot with error bars
        plt.errorbar(thresholds, warning_time_avg_ms, yerr=warning_time_std_ms, label=experiment.name, fmt='o-', capsize=5)

    if min_threshold is None:
        min_threshold = min(thresholds)
    if max_threshold is None:
        max_threshold = max(thresholds)
    if min_warning_time is None:
        min_warning_time = 0
    if max_warning_time is None:
        max_warning_time = 500

    # Put a line at the required warning time
    plt.plot([min_threshold, max_threshold], [required_warning_time*1000, required_warning_time*1000], 'k--')

    # Make the x axis logarithmic
    plt.xscale('log')

    plt.xlim([min_threshold, max_threshold])
    plt.ylim([min_warning_time, max_warning_time])

    plt.xlabel('Threshold')
    if method == 'median':
        plt.ylabel('Median Warning Time [ms]')
    elif method == 'average':
        plt.ylabel('Average Warning Time [ms]')

    plt.title(f'Warning Time vs. Threshold')

    plt.legend()

    plt.show()

def plot_threshold_vs_fpr(experiment_list:list[Experiment], horizon=None, required_warning_time=MINIMUM_WARNING_TIME, min_threshold=None, max_threshold=None, min_warning_time=None, max_warning_time=None, cutoff_far=None, method='median'):
    """ Collected over all shots
    """

    plt.figure()

    for experiment in experiment_list:
        thresholds, false_positive_rate, _ = experiment.false_positive_rate_vs_threshold(horizon=horizon, required_warning_time=required_warning_time, method=method)

        # Plot with error bars
        plt.plot(false_positive_rate, thresholds, label=experiment.name)

    if min_threshold is None:
        min_threshold = min(thresholds)
    if max_threshold is None:
        max_threshold = max(thresholds)

    plt.ylim([min_threshold, max_threshold])
    plt.xlim([0, 1])

    plt.xlabel('FPR')
    plt.ylabel('Threshold')
    
    plt.title(f'Threshold vs. FPR')

    plt.legend()

    plt.show()

def plot_fpr_vs_threshold(experiment_list:list[Experiment], horizon=None, required_warning_time=MINIMUM_WARNING_TIME, min_threshold=None, max_threshold=None, min_warning_time=None, max_warning_time=None, cutoff_far=None, method='median'):
    """ Collected over all shots
    """

    plt.figure()

    for experiment in experiment_list:
        thresholds, false_positive_rate, _ = experiment.false_positive_rate_vs_threshold(horizon=horizon, required_warning_time=required_warning_time, method=method)

        # Plot with error bars
        plt.plot(thresholds, false_positive_rate, label=experiment.name)

    if min_threshold is None:
        min_threshold = min(thresholds)
    if max_threshold is None:
        max_threshold = max(thresholds)

    # Put a line at the required warning time
    #plt.plot([min_threshold, max_threshold], [required_warning_time*1000, required_warning_time*1000], 'k--')

    plt.xlim([min_threshold, max_threshold])
    plt.ylim([0, 1])

    plt.ylabel('FPR')
    plt.xlabel('Threshold')
    
    plt.title(f'FPR vs. Threshold')

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

def plot_risk_compare_models(experiment_list:list[Experiment], shot):
    """Assumes the particular shot's data is the same for all experiments"""

    plt.figure()

    times = experiment_list[0].get_time(shot) * 1000
    final_time = times[-1]
    disruptive = (shot in experiment_list[0].get_disruptive_shot_list())

    # Make a list of easy to see colors for each experiment
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    for i, experiment in enumerate(experiment_list):
        color = colors[i % len(colors)]
        risk = experiment.get_predictor_risk(shot)
        plt.plot(times, risk, label=experiment.name, color=color)
        trained_warning_time = experiment.predictor.trained_required_warning_time*1000
        # Plot with a solid vertical line
        plt.axvline(x=final_time-trained_warning_time, color=color, linestyle='-')
        trained_disruptive_window = experiment.predictor.trained_disruptive_window*1000
        plt.axvline(x=final_time-trained_disruptive_window, color=color, linestyle='--')
    pass

    plt.xlim([times[0], times[-1]])
    plt.ylim([0, 1])

    plt.xlabel('Time [ms]')
    plt.ylabel('Disruption Risk')

    plt.legend()

    if disruptive:
        plt.title(f'Disruption Risk vs. Time for Shot {shot} (Disrupted)')
    else:
        plt.title(f'Disruption Risk vs. Time for Shot {shot} (Not Disrupted)')

def plot_ettd_compare_models(experiment_list:list[Experiment], shot):
    """ NOT RIGOROUS """

    plt.figure()

    times = experiment_list[0].get_time(shot) * 1000
    disruptive = (shot in experiment_list[0].get_disruptive_shot_list())

    # Make a list of easy to see colors for each experiment
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    for i, experiment in enumerate(experiment_list):
        color = colors[i % len(colors)]
        lifetime_curve = make_shot_lifetime_curve(times, disruptive, experiment.predictor.trained_disruptive_window*1000)
        plt.plot(times, lifetime_curve, color=color, linestyle='--')
        ettd_ms = experiment.get_predictor_ettd(shot)*1000
        plt.plot(times, ettd_ms, label=experiment.name, color=color, linestyle='-')

    plt.xlim([times[0], times[-1]])
    #plt.gca().set_ylim(bottom=0)
    plt.ylim([0, 1000])

    plt.xlabel('Time [ms]')
    plt.ylabel('Expected Time To Disruption [ms]')

    plt.legend()

    if disruptive:
        plt.title(f'Expected Time To Disruption vs. Time for Shot {shot} (Disrupted)')
    else:
        plt.title(f'Expected Time To Disruption vs. Time for Shot {shot} (Not Disrupted)')

# Plots for showing the composition of the dataset

def plot_all_shot_durations(experiment:Experiment):

    shot_durations_ms = experiment.get_all_shot_durations()*1000

    plt.figure()

    plt.hist(shot_durations_ms, bins=50)

    plt.xlabel('Shot Duration [ms]')
    plt.ylabel('Number of Shots')

    plt.title(f'Shot Durations for {experiment.name}')

    plt.show()

def plot_disruptive_vs_non_disruptive_shot_durations(experiment:Experiment):

    disruptive_shot_durations_ms = experiment.get_disruptive_shot_durations()*1000
    non_disruptive_shot_durations_ms = experiment.get_non_disruptive_shot_durations()*1000

    plt.figure()

    plt.hist(non_disruptive_shot_durations_ms, bins=50, label='Non-Disruptive')
    plt.hist(disruptive_shot_durations_ms, bins=50, label='Disruptive')

    plt.xlim([min(min(disruptive_shot_durations_ms), min(non_disruptive_shot_durations_ms)), max(max(disruptive_shot_durations_ms), max(non_disruptive_shot_durations_ms))])

    plt.xlabel('Shot Duration [ms]')
    plt.ylabel('Number of Shots')
    plt.title(f'Shot Durations for {experiment.dataset_path}')
    plt.legend()
    plt.show()

