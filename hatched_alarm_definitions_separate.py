import matplotlib.pyplot as plt
from disruption_survival_analysis.Experiments import Experiment

# Using only the output of the random forest model for this example.

LEGEND_FONT_SIZE = 12
TICK_FONT_SIZE = 18
LABEL_FONT_SIZE = 22
TITLE_FONT_SIZE = 24

THRESHOLD = 0.15

def plot_signals(experiment, shot_number, axes):
    shot_data = experiment.get_shot_data(shot_number)
    # Get the predicted risk for this shot
    risk = experiment.get_predictor_risk(shot_number)

    # plot risk
    axes[0].plot(shot_data['time'], risk, label='Risk', color='black')
    axes[0].set_ylim(0, 0.3)
    axes[0].set_yticks([0, THRESHOLD])
    axes[0].set_yticklabels(['0', f"{THRESHOLD:.2f}"])

    # Plot signals
    raw_data = experiment.get_raw_data(shot_number)

    ip_signal = abs(raw_data['ip'])/1e6
    axes[1].plot(raw_data['time'], ip_signal, label='Ip [MA]', color='purple')
    axes[1].set_ylim(0, 1.2)

    ng_signal = abs(raw_data['Greenwald_fraction'])
    axes[2].plot(raw_data['time'], ng_signal, label='Greenwald Fraction', color='purple')
    axes[2].set_ylim(0, 1.2)

    radf_signal = abs(raw_data['radiated_fraction'])
    axes[3].plot(raw_data['time'], radf_signal, label='Radiated Fraction', color='purple')
    axes[3].set_ylim(0, 1.5)

    b_1_norm = abs(raw_data['n_equal_1_normalized'])
    axes[4].plot(raw_data['time'], b_1_norm, label='Normalized n=1', color='purple')
    axes[4].set_ylim(0, 0.002)

def find_important_times(experiment, shot_number, threshold):
    shot_data = experiment.get_shot_data(shot_number)
    # Get the predicted risk for this shot
    risk = experiment.get_predictor_risk(shot_number)
    # Find the first time point where the risk exceeds the threshold
    try:
        trigger_time = shot_data['time'][risk > threshold].values[0]
    except IndexError:
        trigger_time = None

    # Find the time of the disruption
    end_time = shot_data['time'][-1]
    return end_time, trigger_time

def true_positive_hatched(experiment:Experiment):
    # Set up the figure and axis

    PLOT_STYLE = 'seaborn-v0_8-poster'
    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    
    # Create a figure for a true positive
    # Set up the figure and axis
    shot_number = 1150820011 # GOOD FOR TRUE POSITIVE, THRESHOLD = 0.15
    # Get the data for this shot

    fig, axes = plt.subplots(5, 1, figsize=(16, 8), sharex=True)
    #plt.grid(True, color='w', linestyle='-', linewidth=1.5)
    for ax in axes:
        ax.set_facecolor('0.95')

    plt.subplots_adjust(wspace=0, hspace=0)
    
    # 1. plot signals

    plot_signals(experiment, shot_number, axes)

    # 2. disruption time and required warning time

    disruption_time, trigger_time = find_important_times(experiment, shot_number, THRESHOLD)
    required_warning_time = disruption_time-0.2

    # Put a vertical line at the last time point on all axes
    # Also shade in the region left of the line blue
    for ax in axes:
        ax.axvline(disruption_time-.2, color='red', linestyle='--', label='Required Warning Time')
        ax.fill_betweenx(ax.get_ylim(), 0, required_warning_time, color='blue', alpha=0.1)
        ax.fill_betweenx(ax.get_ylim(), required_warning_time, disruption_time, color='red', alpha=0.1)

    # 3. threshold and alarm time
        
    # On the first axis, plot the threshold as 0.15
    axes[0].axhline(THRESHOLD, color='orange', linestyle='--', label='Risk Threshold')
    trigger_time = 0.8

    # 4. hatching for warning time

    # For each axis, hatch the area between the trigger time and the required warning time
    for ax in axes:
        ax.fill_between(ax.get_xlim(), trigger_time, required_warning_time, color='blue', alpha=0.1, hatch='//')

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


