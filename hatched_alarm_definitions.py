import matplotlib.pyplot as plt
from disruption_survival_analysis.Experiments import Experiment

# Using only the output of the random forest model for this example.

def plot_signals(experiment, shot_number, ax):
    shot_data = experiment.get_raw_data(shot_number)
    ax2 = ax.twinx()
    ip_signal = abs(shot_data['ip'])
    ax2.plot(shot_data['time'], ip_signal/max(ip_signal), label='Normalized Ip', color='blue')
    ng_signal = abs(shot_data['Greenwald_fraction'])
    ax2.plot(shot_data['time'], ng_signal, label='Greenwald Fraction', color='green')
    radf_signal = abs(shot_data['radiated_fraction'])
    ax2.plot(shot_data['time'], radf_signal, label='Radiated Fraction', color='red')
    b_1_norm = abs(shot_data['n_equal_1_normalized'])*800
    ax2.plot(shot_data['time'], b_1_norm, label='Normalized n=1', color='purple')
    ax2.set_ylim(0, 1.2)

def true_positive_hatched(ax, experiment:Experiment):
    # Create a figure for a true positive
    # Set up the figure and axis
    shot_number = 1150820011 # GOOD FOR TRUE POSITIVE, THRESHOLD = 0.15
    # Get the data for this shot
    shot_data = experiment.get_shot_data(shot_number)
    
    # Get the predicted risk for this shot
    risk = experiment.get_predictor_risk(shot_number)

    # Plot the risk over time on this axis
    ax.plot(shot_data['time'], risk, label='Risk', color='black')
    ax.set_ylim(0, 0.2)
    plot_signals(experiment, shot_number, ax)

def false_positive_hatched(ax, experiment:Experiment):

    shot_number = 1140515021 # GOOD FOR FALSE POSITIVE, THRESHOLD = 0.15

    # Get the data for this shot
    shot_data = experiment.get_shot_data(shot_number)
    # Get the predicted risk for this shot
    risk = experiment.get_predictor_risk(shot_number)

    # Plot the risk over time on this axis
    ax.plot(shot_data['time'], risk, label='Risk', color='black')
    plot_signals(experiment, shot_number, ax)

def false_negative_hatched(ax, experiment:Experiment):

    shot_number = 1120203005 # GOOD FOR FALSE NEGATIVE, THRESHOLD = 0.15

    # Get the data for this shot
    shot_data = experiment.get_shot_data(shot_number)
    # Get the predicted risk for this shot
    risk = experiment.get_predictor_risk(shot_number)

    # Plot the risk over time on this axis
    ax.plot(shot_data['time'], risk, label='Risk', color='black')
    # Plot the density over time on this axis
    plot_signals(experiment, shot_number, ax)

def true_negative_hatched(ax, experiment:Experiment):

    shot_number = experiment.get_non_disruptive_shot_list()[0]

    shot_data = experiment.get_shot_data(shot_number)

    # Get the predicted risk for this shot
    risk = experiment.get_predictor_risk(shot_number)

    # Plot the risk over time on this axis, with a line width of 3
    ax.plot(shot_data['time'], risk, label='Risk', color='black', linewidth=10)

    plot_signals(experiment, shot_number, ax)
    
def plot_hatched_alarm_definitions(experiment:Experiment):
    LEGEND_FONT_SIZE = 12
    TICK_FONT_SIZE = 18
    LABEL_FONT_SIZE = 22
    TITLE_FONT_SIZE = 24
    # Set up the figure and axis

    PLOT_STYLE = 'seaborn-v0_8-poster'
    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    axislist = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax in axislist:
        # Plot a horizontal line at the threshold
        ax.axhline(0.15, color='red', linestyle='--', label='Risk Threshold')

    # Upper Left Plot: True Positive
    true_positive_hatched(axes[0, 0], experiment)
    false_positive_hatched(axes[0, 1], experiment)
    false_negative_hatched(axes[1, 0], experiment)
    true_negative_hatched(axes[1, 1], experiment)

    

    for ax in axislist:
        ax.set_xlabel('Time [s]', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('Risk', fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        ax.set_ylim(0, 0.3)


    # Adjust layout for better spacing
    plt.tight_layout()


