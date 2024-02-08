# This script generates all figures for the paper in one go. Uses .eps at high resolution for publication.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py
import dill
import os
from disruption_survival_analysis.plot_experiments import save_fig, plot_warning_time_distribution
from disruption_survival_analysis.sweep_config import create_bootstrap_list
from disruption_survival_analysis.experiment_utils import area_under_curve
FIG_PREFIX = 'paper_figures/Fig'

# -----------
# INTRO PLOTS
# -----------

def binary_labeling(fig_count):

     # Binary Labeling
     PLOT_STYLE = 'seaborn-v0_8-poster'


     LEGEND_FONT_SIZE = 12
     TICK_FONT_SIZE = 22
     LABEL_FONT_SIZE = 32
     TITLE_FONT_SIZE = 28

     plt.style.use(PLOT_STYLE)
     plt.rc('text', usetex=True)
     plt.rc('font', family='serif')

     # Create a figure and axis
     fig, (ax2, ax1) = plt.subplots(ncols=2, figsize=(21, 4))

     # Example data: time slices and corresponding binary labels
     time_slices = [2, 3, 4, 5, 6, 7, 8, 9]
     binary_labels = [0, 0, 0, 0, 0, 1, 1, 1]

     # Plotting vertical lines for each time slice
     for time_slice, label in zip(time_slices, binary_labels):
          ax1.axvline(x=time_slice, color='black', linestyle='--', alpha=0.7)
          # Shade 0 regions in green and 1 regions in red
          if label == 0:
               ax1.axvspan(time_slice, time_slice + 1, color='green', alpha=0.3)
               ax1.text(time_slice + 0.5, 0.5, r'$ND$', fontsize=LABEL_FONT_SIZE, horizontalalignment='center', verticalalignment='center')
          else:
               ax1.axvspan(time_slice, time_slice + 1, color='red', alpha=0.3)
               ax1.text(time_slice + 0.5, 0.5, r'$D$', fontsize=LABEL_FONT_SIZE, horizontalalignment='center', verticalalignment='center')

     # Set plot title and labels
     ax1.set_title('Binary Classifier Disruptive Shot Labeling', fontsize=TITLE_FONT_SIZE)
     ax1.set_xlim(2, 10)

     # Remove y ticks
     ax1.set_yticks([])

     # Put a tick at the 6th time slice that has the text "t"
     ax1.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10], ["", "", "", "", "", r'$t_{\mathrm{disrupt}} - \Delta\tau_{\mathrm{class}}$', "", "", r'$t_{\mathrm{disrupt}}$'], fontsize=TICK_FONT_SIZE+8)


     plt.style.use(PLOT_STYLE)
     plt.rc('text', usetex=True)
     plt.rc('font', family='serif')

     # Example data: time slices and corresponding binary labels
     time_slices = [2, 3, 4, 5, 6, 7, 8, 9]
     binary_labels = [0, 0, 0, 0, 0, 0, 0, 0]

     # Plotting vertical lines for each time slice
     for time_slice, label in zip(time_slices, binary_labels):
     
          # Shade 0 regions in green and 1 regions in red
          if label == 0:
               ax2.axvspan(time_slice, time_slice + 1, color='green', alpha=0.3)
               ax2.text(time_slice + 0.5, 0.5, r'$ND$', fontsize=LABEL_FONT_SIZE, horizontalalignment='center', verticalalignment='center')
          else:
               ax2.axvspan(time_slice, time_slice + 1, color='red', alpha=0.3)
               ax2.text(time_slice + 0.5, 0.5, r'$D$', fontsize=LABEL_FONT_SIZE, horizontalalignment='center', verticalalignment='center')
          ax2.axvline(x=time_slice, color='black', linestyle='--', alpha=0.7)

     # Set plot title and labels
     ax2.set_title('Binary Classifier Non-Disruptive Shot Labeling', fontsize=TITLE_FONT_SIZE)
     #plt.xlabel('Time Slices', fontsize=LABEL_FONT_SIZE)
     ax2.set_xlim(2, 10)

     # Remove y ticks
     ax2.set_yticks([])

     # Make all x ticks blank
     ax2.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10], ["", "", "", "", "", "", "", "", r'$t_{\mathrm{end}}$'], fontsize=TICK_FONT_SIZE+8)

     plt.subplots_adjust(wspace=0.1)

     save_fig(fig, f"{FIG_PREFIX}{fig_count}")
     plt.rcParams.update(mpl.rcParamsDefault)

def kaplan_meier(fig_count):
     # Generate random data points
     np.random.seed(0)
     x_old = [0.1, 0.2, 0.3, 0.4]
     x_fit_points = [0.5, 0.6, 0.7, 0.8, 0.9]

     y_old = [0.1, 0.3, 0.1, 0.2]
     y_fit_points = [0.2, 0.4, 0.5, 0.45, 0.4]

     # Fit a linear regression model
     coefficients = np.polyfit(x_fit_points, y_fit_points, 1)
     p = np.poly1d(coefficients)

     # Generate points for the line of best fit
     x_fit = np.linspace(0.5, 0.9, 100)
     y_fit = p(x_fit)

     # Extrapolation
     x_extrapolation = np.linspace(0.9, 1.3, 50)
     y_extrapolation = x_extrapolation*coefficients[0]-0.01

     fig = plt.figure()

     plt.style.use('seaborn-v0_8-poster')

     plt.rc('text', usetex=True)
     plt.rc('font', family='serif')

     # Plot the data points, line of best fit, and extrapolation
     plt.scatter((x_fit_points + x_old), (y_fit_points + y_old), color='black', label='RF Output')
     plt.fill_between(np.linspace(0.5, 0.9, 100), 1, alpha=0.15, color='green', label='Fitting Window')
     plt.plot(x_fit, y_fit, label='Line of Best Fit', color='blue', linewidth=4)
     plt.plot(x_extrapolation, y_extrapolation, color='red', linestyle='--', linewidth=4, label='Extrapolation')



     plt.xlabel('Time [s]', fontsize=26)
     plt.ylabel('Disruptive Class Prediction', fontsize=26)

     # Increase tick size
     plt.xticks(fontsize=20)
     plt.yticks(fontsize=20)


     plt.ylim((0, 1))
     plt.xlim((0, 1.5))


     plt.legend(loc='upper left', fontsize=14)
     save_fig(fig, f"{FIG_PREFIX}{fig_count}")

     plt.rcParams.update(mpl.rcParamsDefault)

def sr_labeling(fig_count):

     PLOT_STYLE = 'seaborn-v0_8-poster'
     import matplotlib as mpl

     LEGEND_FONT_SIZE = 12
     TICK_FONT_SIZE = 18
     LABEL_FONT_SIZE = 22
     TITLE_FONT_SIZE = 24

     plt.style.use(PLOT_STYLE)
     plt.rc('text', usetex=True)
     plt.rc('font', family='serif')

     # Create a figure and axis
     fig, (ax2, ax1) = plt.subplots(ncols=2, figsize=(21, 4))

     # Example data: time slices and corresponding binary labels
     time_slices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
     binary_labels = [0, 0, 0, 0, 0, 0, 0, 0]

     # Plotting vertical lines for each time slice
     for time_slice, label in zip(time_slices, binary_labels):
          ax1.axvline(x=time_slice, color='black', linestyle='--', alpha=0.7)
          # Shade 0 regions in green and 1 regions in red
          ax1.axvspan(time_slice, time_slice + 1, color='orange', alpha=0.1)
          ax1.text(time_slice + 0.5, 0.7, r'$t = $' + f" {7 - time_slice}", fontsize=LABEL_FONT_SIZE, horizontalalignment='center', verticalalignment='center')
          ax1.text(time_slice + 0.5, 0.3, r'$\delta = 1$', fontsize=LABEL_FONT_SIZE, horizontalalignment='center', verticalalignment='center')

     # Set plot title and labels
     ax1.set_title('Survival Regression Disruptive Shot Labeling', fontsize=TITLE_FONT_SIZE)
     ax1.set_xlim(0, 8)

     # Remove y ticks
     ax1.set_yticks([])

     ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ["", "", "", "", "", "", "", "", r"$t_{\mathrm{disrupt}}$"], fontsize=TICK_FONT_SIZE+8)

     # Example data: time slices and corresponding binary labels
     time_slices = [0, 1, 2, 3, 4, 5, 6, 7, 8]

     # Plotting vertical lines for each time slice
     for time_slice, label in zip(time_slices, binary_labels):
          ax2.axvline(x=time_slice, color='black', linestyle='--', alpha=0.7)
          # Shade 0 regions in green and 1 regions in red
          ax2.axvspan(time_slice, time_slice + 1, color='blue', alpha=0.1)
          ax2.text(time_slice + 0.5, 0.7, r'$t = $' + f" {7 - time_slice}", fontsize=LABEL_FONT_SIZE, horizontalalignment='center', verticalalignment='center')
          ax2.text(time_slice + 0.5, 0.3, r'$\delta = 0$', fontsize=LABEL_FONT_SIZE, horizontalalignment='center', verticalalignment='center')

     # Set plot title and labels
     ax2.set_title('Survival Regression Non-Disruptive Shot Labeling', fontsize=TITLE_FONT_SIZE)
     ax2.set_xlim(0, 8)

     # Remove y ticks
     ax2.set_yticks([])

     ax2.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ["", "", "", "", "", "", "", "", r"$t_{\mathrm{end}}$"], fontsize=TICK_FONT_SIZE+8)

     plt.subplots_adjust(wspace=0.1)

     save_fig(fig, f"{FIG_PREFIX}{fig_count}")
     plt.rcParams.update(mpl.rcParamsDefault)

def alarm_definitions(fig_count):
     LEGEND_FONT_SIZE = 12
     TICK_FONT_SIZE = 18
     LABEL_FONT_SIZE = 22
     TITLE_FONT_SIZE = 24
     # Set up the figure and axis

     PLOT_STYLE = 'seaborn-v0_8-poster'
     plt.style.use(PLOT_STYLE)
     plt.rc('text', usetex=True)
     plt.rc('font', family='serif')

     fig, axes = plt.subplots(2, 2, figsize=(10, 4))

     # Remove y-axis for all plots
     for ax in axes.flatten():
          ax.get_yaxis().set_visible(False)

     # Upper Left Plot: True Positive
     axes[0, 0].set_title('True Positive')
     axes[0, 0].axvline(x=0.7, color='red')
     axes[0, 0].axvline(x=2.5, ymax=0.49, color='blue', linestyle='--')
     axes[0, 0].axvspan(2.5, 6, ymax=0.49, color='blue', alpha=0.1)
     axes[0, 0].axvspan(0.7, 6, ymin=0.50, color='red', alpha=0.1)
     axes[0, 0].set_xticks([0, 0.7, 2.5, 6])
     axes[0, 0].set_xticklabels(['', r'$t_\textrm{alarm}$', r'$t_\textrm{disrupt} - \Delta t_\textrm{req}$', r'$t_\textrm{disrupt}$'])
     axes[0, 0].set_xlim(0, 6)
     axes[0, 0].text(3.4, 0.75, r'$\Delta t_\textrm{warn}$', fontsize=TICK_FONT_SIZE, horizontalalignment='center', verticalalignment='center')
     axes[0, 0].text(4.3, 0.23, r'$\Delta t_\textrm{req}$', fontsize=TICK_FONT_SIZE, horizontalalignment='center', verticalalignment='center')

     # Upper Right Plot: False Positive
     axes[0, 1].set_title('False Positive')
     axes[0, 1].axvline(x=3, color='red')
     axes[0, 1].set_xticks([0, 3, 6])
     axes[0, 1].set_xticklabels(['', r'$t_\textrm{alarm}$', r'$t_\textrm{end}$'])
     axes[0, 1].set_xlim(0, 6)

     # Lower Left Plot: False Negative
     axes[1, 0].set_title('False Negative')
     axes[1, 0].axvline(x=4.5, color='red')
     axes[1, 0].axvline(x=2.5, ymax=0.49, color='blue', linestyle='--')
     axes[1, 0].axvspan(2.5, 6, ymax=0.49, color='blue', alpha=0.1)
     axes[1, 0].axvspan(4.5, 6, ymin=0.50, color='red', alpha=0.1)
     axes[1, 0].set_xticks([0, 2.5, 4.5, 6])
     axes[1, 0].set_xticklabels(['', r'$t_\textrm{disrupt} - \Delta t_\textrm{req}$', r'$t_\textrm{alarm}$', r'$t_\textrm{disrupt}$'])
     axes[1, 0].set_xlim(0, 6)
     axes[1, 0].text(5.3, 0.75, r'$\Delta t_\textrm{warn}$', fontsize=TICK_FONT_SIZE, horizontalalignment='center', verticalalignment='center')
     axes[1, 0].text(3.8, 0.23, r'$\Delta t_\textrm{req}$', fontsize=TICK_FONT_SIZE, horizontalalignment='center', verticalalignment='center')

     # Lower Right Plot: True Negative
     axes[1, 1].set_title('True Negative')
     axes[1, 1].set_xticks([0, 6])
     axes[1, 1].set_xticklabels(['', r'$t_\textrm{end}$'])
     axes[1, 1].set_xlim(0, 6)

     # Adjust layout for better spacing
     plt.tight_layout()

     save_fig(fig, f"{FIG_PREFIX}{fig_count}")

     plt.rcParams.update(mpl.rcParamsDefault)

def auwtc(fig_count):
     LEGEND_FONT_SIZE = 12
     TICK_FONT_SIZE = 22
     LABEL_FONT_SIZE = 24
     TITLE_FONT_SIZE = 26
     
     PLOT_STYLE = 'seaborn-v0_8-poster'
     plt.style.use(PLOT_STYLE)
     plt.rc('text', usetex=True)
     plt.rc('font', family='serif')

     plt.grid(True, color='w', linestyle='-', linewidth=1.5)
     plt.gca().patch.set_facecolor('0.92')

     fig = plt.figure()

     # Create some example data
     x = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
     y = [0, 10, 25, 30, 40, 78, 90, 100, 120]

     plt.plot(x, y, linewidth=8)

     plt.xlim([0, 0.05])
     plt.ylim([0, 120])

     plt.xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
               ["0\%", "1\%", "2\%", "3\%", "4\%", "5\%", "6\%", "7\%", "8\%"])

     # Make ticks bigger
     plt.xticks(fontsize=TICK_FONT_SIZE+8)
     plt.yticks(fontsize=TICK_FONT_SIZE+8)

     #plt.legend(fontsize=LEGEND_FONT_SIZE)
     plt.xlabel("FPR", fontsize=LABEL_FONT_SIZE+8)
     plt.ylabel("IQM Warning Time [ms]", fontsize=LABEL_FONT_SIZE+8)

     # Put a vertical line at 0.05 false alarm rate
     plt.axvline(x=0.05, color='orange', linestyle='--', linewidth=6)

     # Shade in the region below the line
     plt.fill_between(x[:6], 0, y[:6], color='green', alpha=0.15)

     # Put a horizontal line at the required warning time
     #plt.axhline(y=required_warning_time*1000, color='k', linestyle='--')

     plt.title("IQM Warning Time vs. FPR", fontsize=TITLE_FONT_SIZE+8)
     
     save_fig(fig, f"{FIG_PREFIX}{fig_count}")

     plt.rcParams.update(mpl.rcParamsDefault)

def sthr(fig_count):
     LEGEND_FONT_SIZE = 12
     TICK_FONT_SIZE = 22
     LABEL_FONT_SIZE = 24
     TITLE_FONT_SIZE = 26
     
     PLOT_STYLE = 'seaborn-v0_8-poster'
     plt.style.use(PLOT_STYLE)
     plt.rc('text', usetex=True)
     plt.rc('font', family='serif')

     plt.grid(True, color='w', linestyle='-', linewidth=1.5)
     plt.gca().patch.set_facecolor('0.92')

     fig = plt.figure()

     # Create some example data
     x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
     y = [0.1, 0.1, 0.3, 0.4, 0.8, 0.6, 0.9, 0.5, 0.5]

     plt.plot(x, y, linewidth=8)

     plt.xlim([0, 8])
     plt.ylim([0, 1])

     # Make ticks bigger
     plt.yticks(fontsize=TICK_FONT_SIZE+8)

     #plt.legend(fontsize=LEGEND_FONT_SIZE)
     #plt.xlabel("Time [s]", fontsize=LABEL_FONT_SIZE+8)
     plt.ylabel("Predicted Risk", fontsize=LABEL_FONT_SIZE+8)

     # Put a horizontal line at 0.5 risk
     plt.axhline(y=0.6, color='orange', linestyle='--', linewidth=6)

     # Put a vertical line at 3.5 seconds
     plt.axvline(x=3.5, color='red', linewidth=6)

     # Label the x ticks
     plt.xticks([0, 1, 2, 3, 3.5, 4, 5, 6, 7, 8],
               ["", "", "", "", r'$t_\textrm{alarm}$', "", "", "", "", ""], fontsize=TICK_FONT_SIZE+8)

     # Put a horizontal line at the required warning time
     #plt.axhline(y=required_warning_time*1000, color='k', linestyle='--')

     plt.title("Simple Threshold Alarm", fontsize=TITLE_FONT_SIZE+8)

     save_fig(fig, f"{FIG_PREFIX}{fig_count}")

     plt.rcParams.update(mpl.rcParamsDefault)

def warning_time_distribution(fig_count):
     # Load experiment_groups from disk
     from disruption_survival_analysis.sweep_config import create_experiment_groups, get_experiments
     from disruption_survival_analysis.plot_experiments import plot_restricted_mean_survival_time_shot
     from disruption_survival_analysis.manage_datasets import load_disruptive_shot_list, load_non_disruptive_shot_list

     #device = 'synthetic'
     #dataset_path = 'test'
     devices = ['cmod']
     #dataset_paths = ['preliminary_dataset_no_ufo']
     #dataset = 'sql_all_no_ufo'
     dataset='paper_4'
     dataset_paths = [f"{dataset}/stack_10"]

     # models, alarms, metrics, and minimum warning times to use
     models = ['cph']
     alarms = ['sthr']
     metrics = ['auroc']
     min_warning_times = [0.01]

     experiment_groups = create_experiment_groups(devices, dataset_paths, models, alarms, metrics, min_warning_times)

     experiment_list = get_experiments(experiment_groups, ['auroc'])
     
     plot_warning_time_distribution(experiment_list[0], 0.05, save=f"{FIG_PREFIX}{fig_count}")

# ------------------------------
# BOOTSTRAP CONFIDENCE INTERVALS
# ------------------------------

def bootstrap_auroc(fig_count):
     sets = []
     all_alarms = ['sthr']
     all_warnings = [0.01, 0.05, 0.1]
     warn_colors = ['red', 'orange', 'yellow', 'blue', 'purple', 'cyan']
     dataset_name = "paper_4"
     dataset_path = f"{dataset_name}/stack_10"
     all_models = ['rf', 'km', 'cph', 'dcph', 'dsm']

     for model in all_models:
          sets.append({'models': [model],
                         'alarms': all_alarms,
                         'warnings': all_warnings,
                         "title": f"{model}"})
     
     PLOT_STYLE = 'seaborn-v0_8-poster'
     plt.style.use(PLOT_STYLE)
     plt.rc('text', usetex=True)
     plt.rc('font', family='serif')
     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
     
     ax1.grid(True, color='w', linestyle='-', linewidth=1.5)
     ax2.grid(True, color='w', linestyle='-', linewidth=1.5)
     #plt.gca().patch.set_facecolor('0.92')
     #plt.gca().set_axisbelow(True)
     ax1.patch.set_facecolor('0.92')
     ax2.patch.set_facecolor('0.92')
     ax1.set_axisbelow(True)
     ax2.set_axisbelow(True)

     x = np.arange(len(sets))
     width = 0.12

     device = 'cmod'
     max_area = 1.0
     title = "C-Mod AUROC"

     for i, set in enumerate(sets):
          models = set['models']
          alarms = set['alarms']
          min_warning_times = set['warnings']
          bootstrap_list = create_bootstrap_list([device], [dataset_path], models, alarms, ["auroc", "auwtc"], min_warning_times)

          for j, bootstrapped_metrics in enumerate(bootstrap_list):
               unique_fars = bootstrapped_metrics['fars']
               offset = (j-2.5)*width

               upper_tars = bootstrapped_metrics['upper_tars']
               lower_tars = bootstrapped_metrics['lower_tars']
               #avg_tars = bootstrapped_metrics['mean_tars']
               med_tars = bootstrapped_metrics['median_tars']
               upper_area = area_under_curve(unique_fars, upper_tars)
               lower_area = area_under_curve(unique_fars, lower_tars)
               typ_area = area_under_curve(unique_fars, med_tars)

               if i == 0 and j < 3:
                    ax1.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"ROC {min_warning_times[j]*1000} ms")
               elif i == 0 and j >= 3:
                    ax1.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"WTC {min_warning_times[j-3]*1000} ms")
               else:
                    ax1.bar(x[i]+offset, typ_area, width, color=warn_colors[j])
               upper_error_bar = max(0, upper_area - typ_area)
               lower_error_bar = max(0, typ_area - lower_area)
               ax1.errorbar(x[i]+offset, typ_area, yerr=[[lower_error_bar], [upper_error_bar]], fmt='', ecolor='k', capsize=10)

     if device == 'cmod':
          ax1.legend(loc='upper left', fontsize=13, ncol=2)
     else:
          #plt.legend(loc='upper left', fontsize=16)
          pass

     ax1.set_xticks(x, [set['title'].upper() for set in sets])
     ax1.set_ylabel(f"Bootstrap AUROC")
     #plt.xlabel("MODEL")
     ax1.set_ylim([0.4, max_area])

     ax1.set_title(title)

     device = 'd3d'
     max_area = 1.0
     title = "DIII-D AUROC"

     for i, set in enumerate(sets):
          models = set['models']
          alarms = set['alarms']
          min_warning_times = set['warnings']
          bootstrap_list = create_bootstrap_list([device], [dataset_path], models, alarms, ["auroc", "auwtc"], min_warning_times)

          for j, bootstrapped_metrics in enumerate(bootstrap_list):
               unique_fars = bootstrapped_metrics['fars']
               offset = (j-2.5)*width

               upper_tars = bootstrapped_metrics['upper_tars']
               lower_tars = bootstrapped_metrics['lower_tars']
               #avg_tars = bootstrapped_metrics['mean_tars']
               med_tars = bootstrapped_metrics['median_tars']
               upper_area = area_under_curve(unique_fars, upper_tars)
               lower_area = area_under_curve(unique_fars, lower_tars)
               typ_area = area_under_curve(unique_fars, med_tars)

               if i == 0 and j < 3:
                    ax2.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"ROC {min_warning_times[j]*1000} ms")
               elif i == 0 and j >= 3:
                    ax2.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"WTC {min_warning_times[j-3]*1000} ms")
               else:
                    ax2.bar(x[i]+offset, typ_area, width, color=warn_colors[j])
               upper_error_bar = max(0, upper_area - typ_area)
               lower_error_bar = max(0, typ_area - lower_area)
               ax2.errorbar(x[i]+offset, typ_area, yerr=[[lower_error_bar], [upper_error_bar]], fmt='', ecolor='k', capsize=10)

     if device == 'cmod':
          ax2.legend(loc='upper left', fontsize=13, ncol=2)
     else:
          #plt.legend(loc='upper left', fontsize=16)
          pass

     ax2.set_xticks(x, [set['title'].upper() for set in sets])
     ax2.set_ylabel(f"Bootstrap AUROC")
     #plt.xlabel("MODEL")
     ax2.set_ylim([0.4, max_area])

     ax2.set_title(title)

     save_fig(fig, f"{FIG_PREFIX}{fig_count}")

     plt.rcParams.update(mpl.rcParamsDefault)
     
def bootstrap_auwtc(fig_count):

     sets = []
     all_alarms = ['sthr']
     all_warnings = [0.01, 0.05, 0.1]
     warn_colors = ['red', 'orange', 'yellow', 'blue']

     dataset_name = "paper_4"
     dataset_path = f"{dataset_name}/stack_10"
     all_models = ['rf', 'km', 'cph', 'dcph', 'dsm']

     for model in all_models:
          sets.append({'models': [model],
                         'alarms': all_alarms,
                         'warnings': all_warnings,
                         "title": f"{model}"})
     
     PLOT_STYLE = 'seaborn-v0_8-poster'
     plt.style.use(PLOT_STYLE)
     plt.rc('text', usetex=True)
     plt.rc('font', family='serif')
     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

     ax1.grid(True, color='w', linestyle='-', linewidth=1.5)
     ax2.grid(True, color='w', linestyle='-', linewidth=1.5)
     ax1.patch.set_facecolor('0.92')
     ax2.patch.set_facecolor('0.92')
     ax1.set_axisbelow(True)
     ax2.set_axisbelow(True)

     x = np.arange(len(sets))
     width = 0.18

     device = 'cmod'
     max_area = 1.75

     for i, set in enumerate(sets):
          models = set['models']
          alarms = set['alarms']
          min_warning_times = set['warnings']
          full_bootstrap_list = create_bootstrap_list([device], [dataset_path], models, alarms, ["auroc", "auwtc"], min_warning_times)

          # Only keep 50ms AUWTC
          bootstrap_list = []
          for k, item in enumerate(full_bootstrap_list):
               if k in [0, 1, 2, 4]:
                    bootstrap_list.append(item)

          for j, bootstrapped_metrics in enumerate(bootstrap_list):
               unique_fars = bootstrapped_metrics['fars']
               offset = (j-1.5)*width

               upper_warns = bootstrapped_metrics['upper_warns']
               lower_warns = bootstrapped_metrics['lower_warns']
               upper_area = area_under_curve(unique_fars, upper_warns, x_cutoff=0.05)*1000
               lower_area = area_under_curve(unique_fars, lower_warns, x_cutoff=0.05)*1000
               typ_area = area_under_curve(unique_fars, bootstrapped_metrics['median_warns'], x_cutoff=0.05)*1000

               if i == 0 and j < 3:
                    ax1.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"ROC {min_warning_times[j]*1000} ms")
               elif i == 0 and j == 3:
                    ax1.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"WTC All")
               else:
                    ax1.bar(x[i]+offset, typ_area, width, color=warn_colors[j])
               upper_error_bar = max(0, upper_area - typ_area)
               lower_error_bar = max(0, typ_area - lower_area)
               ax1.errorbar(x[i]+offset, typ_area, yerr=[[lower_error_bar], [upper_error_bar]], fmt='', ecolor='k', capsize=10)


     ax1.legend(loc='upper right', fontsize=16)

     ax1.set_xticks(x, [set['title'].upper() for set in sets])
     ax1.set_ylabel(f"Bootstrap AUWTC [ms]")
     ax1.set_ylim([0, max_area])

     ax1.set_title("C-Mod AUWTC")

     device = 'd3d'
     max_area = 16

     for i, set in enumerate(sets):
          models = set['models']
          alarms = set['alarms']
          min_warning_times = set['warnings']
          full_bootstrap_list = create_bootstrap_list([device], [dataset_path], models, alarms, ["auroc", "auwtc"], min_warning_times)

          # Only keep 50ms AUWTC
          bootstrap_list = []
          for k, item in enumerate(full_bootstrap_list):
               if k in [0, 1, 2, 4]:
                    bootstrap_list.append(item)

          for j, bootstrapped_metrics in enumerate(bootstrap_list):
               unique_fars = bootstrapped_metrics['fars']
               offset = (j-1.5)*width

               upper_warns = bootstrapped_metrics['upper_warns']
               lower_warns = bootstrapped_metrics['lower_warns']
               upper_area = area_under_curve(unique_fars, upper_warns, x_cutoff=0.05)*1000
               lower_area = area_under_curve(unique_fars, lower_warns, x_cutoff=0.05)*1000
               typ_area = area_under_curve(unique_fars, bootstrapped_metrics['median_warns'], x_cutoff=0.05)*1000

               if i == 0 and j < 3:
                    ax2.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"ROC {min_warning_times[j]*1000} ms")
               elif i == 0 and j == 3:
                    ax2.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"WTC All")
               else:
                    ax2.bar(x[i]+offset, typ_area, width, color=warn_colors[j])
               upper_error_bar = max(0, upper_area - typ_area)
               lower_error_bar = max(0, typ_area - lower_area)
               ax2.errorbar(x[i]+offset, typ_area, yerr=[[lower_error_bar], [upper_error_bar]], fmt='', ecolor='k', capsize=10)


     #ax2.legend(loc='upper right', fontsize=16)

     ax2.set_xticks(x, [set['title'].upper() for set in sets])
     ax2.set_ylabel(f"Bootstrap AUWTC [ms]")
     ax2.set_ylim([0, max_area])

     ax2.set_title("DIII-D AUWTC")

     save_fig(fig, f"{FIG_PREFIX}{fig_count}")

     plt.rcParams.update(mpl.rcParamsDefault)

#def bootstrap_rmstsl(fig_count):

# ---------------------------------
# EXPECTED TIME TO DISRUPTION PLOTS
# ---------------------------------

def ettd_plots(fig_count):
     # Load experiment_groups from disk
     from disruption_survival_analysis.sweep_config import create_experiment_groups, get_experiments
     from disruption_survival_analysis.plot_experiments import plot_restricted_mean_survival_time_shot
     from disruption_survival_analysis.manage_datasets import load_disruptive_shot_list, load_non_disruptive_shot_list

     devices = ['cmod']
     dataset_path='paper_4/stack_10'
     #dataset_path = 'sql_match/stack_10'

     # models, alarms, metrics, and minimum warning times to use
     models = ['rf', 'km', 'cph', 'dcph', 'dsm']
     alarms = ['sthr']
     model_type = 'rmstsl'
     metrics = [model_type]
     min_warning_times = [0.01]

     # Load models and create experiments
     experiment_groups = create_experiment_groups(devices, [dataset_path], models, alarms, metrics, min_warning_times)

     experiment_list = get_experiments(experiment_groups, 
                                   [dataset_path, 'rf', model_type],
                                   [dataset_path, 'km', model_type],
                                   [dataset_path, 'cph', model_type],
                                   [dataset_path, 'dcph', model_type],
                                   [dataset_path, 'dsm', model_type],)

     #experiment_list = get_experiments(experiment_groups, 
     #                                  [dataset_path, 'dcph', 'auroc'])

     # # load disruptive shot list
     # disruptive_shots = load_non_disruptive_shot_list(devices[0], dataset_path, 'test')

     # # Randomize order of the list
     # import random
     # random.shuffle(disruptive_shots)

     selected_shots = [1150820011, 1140515021, 1120621012, 1120712029]

     #plot_restricted_mean_survival_time_shot(experiment_list, 1120215012)

     # # Plot expected time to disruption for each experiment
     plot_restricted_mean_survival_time_shot(experiment_list, selected_shots[0], save=f"{FIG_PREFIX}{fig_count}")
     fig_count += 1
     plot_restricted_mean_survival_time_shot(experiment_list, selected_shots[1], save=f"{FIG_PREFIX}{fig_count}")
     fig_count += 1
     plot_restricted_mean_survival_time_shot(experiment_list, selected_shots[2], save=f"{FIG_PREFIX}{fig_count}")
     fig_count += 1
     plot_restricted_mean_survival_time_shot(experiment_list, selected_shots[3], save=f"{FIG_PREFIX}{fig_count}")


# --------
# ORDERING
# --------

# binary_labeling(1)
# kaplan_meier(2)
# sr_labeling(3)
# alarm_definitions(4)
# auwtc(5)
# sthr(6)
# warning_time_distribution(7)
# bootstrap_auroc(8)
# bootstrap_auwtc(9)


# ettd_plots(10)



# --------
# DATASETS
# --------
devices = ['cmod', 'd3d']
bootstrap_path = 'paper_4/stack_10/bootstraps'
metrics = ['auroc', 'auwtc']
warning_times_ms = [10, 50, 100]
models = ['rf', 'km', 'cph', 'dcph', 'dsm']
alarms = ['sthr']


def pack_auroc_bootstrap_results(f):
     # Make a new dataset for each bootstrap result
     for device in devices:
          for model in models:
               for alarm in alarms:
                    for metric in metrics:
                         for warn in warning_times_ms:
                              pkl_path = f"results/{device}/{bootstrap_path}/{model}_{alarm}_{metric}_{warn}ms_bootstrap.pkl"
                              # Load AUROC bootstrap results
                              with open(pkl_path, 'rb') as bootstrap_file:
                                   bootstrap_results = dill.load(bootstrap_file)
                                   # obtain the arrays of interest
                                   fars = bootstrap_results['fars']
                                   upper_tars = bootstrap_results['upper_tars']
                                   lower_tars = bootstrap_results['lower_tars']
                                   median_tars = bootstrap_results['median_tars']

                                   names = ["FAR", "Q1 Avg TAR", "Median Avg TAR", "Q3 Avg TAR"]

                                   ds_dt = np.dtype({'names': names, 'formats': ['f8']*len(names)})

                                   rec_arr = np.rec.fromarrays([fars, lower_tars, median_tars, upper_tars], dtype=ds_dt)

                                   dset = f.create_dataset(f"bootstraps/auroc/{device}/{model}_{metric}_{warn}ms", data=rec_arr)


def pack_auwtc_bootstrap_results(f):
     # Make a new dataset for each bootstrap result
     for device in devices:
          for model in models:
               for alarm in alarms:
                    for metric in metrics:
                         for warn in warning_times_ms:
                              pkl_path = f"results/{device}/{bootstrap_path}/{model}_{alarm}_{metric}_{warn}ms_bootstrap.pkl"
                              # Load AUROC bootstrap results
                              
                              with open(pkl_path, 'rb') as bootstrap_file:
                                   bootstrap_results = dill.load(bootstrap_file)
                                   # obtain the arrays of interest
                                   fars = bootstrap_results['fars']
                                   upper_warns = bootstrap_results['upper_warns']
                                   lower_warns = bootstrap_results['lower_warns']
                                   median_warns = bootstrap_results['median_warns']

                                   names = ["FAR", "Q1 IQM Warn", "Median IQM Warn", "Q3 IQM Warn"]

                                   ds_dt = np.dtype({'names': names, 'formats': ['f8']*len(names)})

                                   rec_arr = np.rec.fromarrays([fars, lower_warns, median_warns, upper_warns], dtype=ds_dt)

                                   if metric == 'auwtc' and warn == 50:
                                        dset = f.create_dataset(f"bootstraps/auwtc/{device}/{model}_{metric}_all", data=rec_arr)
                                   else:
                                        dset = f.create_dataset(f"bootstraps/auwtc/{device}/{model}_{metric}_{warn}ms", data=rec_arr)

def pack_rmst_bootstrap_results(f):
     metrics = ['auroc', 'auwtc', 'rmstsl']

     # Make a new dataset for each bootstrap result
     for device in devices:
          for model in models:
               for alarm in alarms:
                    for metric in metrics:
                         for warn in warning_times_ms:
                              pkl_path = f"results/{device}/paper_4/stack_10/squared_last_rmst/{model}_{alarm}_{metric}_{warn}ms/all_rmst_results.pkl"
                              # Load AUROC bootstrap results
                              try:
                                   with open(pkl_path, 'rb') as bootstrap_file:
                                        bootstrap_results = dill.load(bootstrap_file)
                                        # obtain the arrays of interest
                                        disruptive_rmst_diffs = bootstrap_results['disruptive_rmst_diffs']
                                        non_disruptive_rmst_diffs = bootstrap_results['non_disruptive_rmst_diffs']

                                        if metric in ['auwtc', 'rmstsl'] and warn == 10:
                                             dset = f.create_dataset(f"disruptive_rmst_squared_error/{device}/{model}_{metric}_all", data=disruptive_rmst_diffs)
                                             dset = f.create_dataset(f"non_disruptive_rmst_squared_error/{device}/{model}_{metric}_all", data=non_disruptive_rmst_diffs)
                                        else:
                                             dset = f.create_dataset(f"disruptive_rmst_squared_error/{device}/{model}_{metric}_{warn}ms", data=disruptive_rmst_diffs)
                                             dset = f.create_dataset(f"non_disruptive_rmst_squared_error/{device}/{model}_{metric}_{warn}ms", data=non_disruptive_rmst_diffs)
                              except FileNotFoundError:
                                   pass

def pack_shot_duration_results(f):

     pkl_path = f"plots/save_warning_times.pkl"
     with open(pkl_path, 'rb') as file:
          warning_times = dill.load(file)
          # obtain the arrays of interest
          
          dset = f.create_dataset(f"cph_warning_times", data=warning_times)

def pack_rmst_plot_results(f):

     figs = ["Fig10", "Fig11", "Fig12", "Fig13"]
     shots = [1150820011, 1140515021, 1120621012, 1120712029]

     for fig, shot in zip(figs, shots):
          pkl_path = f"plots/paper_figures/{fig}_ettd_lines.pkl"
          with open(pkl_path, 'rb') as file:
               data = dill.load(file)
               # obtain the arrays of interest
               t = data['rf_sthr_rmstsl_10ms']['t']
               RF = data['rf_sthr_rmstsl_10ms']['ettd']
               KM = data['km_sthr_rmstsl_10ms']['ettd']
               CPH = data['cph_sthr_rmstsl_10ms']['ettd']
               DCPH = data['dcph_sthr_rmstsl_10ms']['ettd']
               DSM = data['dsm_sthr_rmstsl_10ms']['ettd']

               names = ["t", "RF", "KM", "CPH", "DCPH", "DSM"]

               ds_dt = np.dtype({'names': names, 'formats': ['f8']*len(names)})

               rec_arr = np.rec.fromarrays([t, RF, KM, CPH, DCPH, DSM], dtype=ds_dt)

               dset = f.create_dataset(f"cmod_rmst_plots/{shot}", data=rec_arr)

# Remove old file if it exists
#if os.path.exists('plots/paper_figures/plot_data.hdf5'):
#     os.remove('plots/paper_figures/plot_data.hdf5')

f = h5py.File('plots/paper_figures/plot_data.hdf5', 'w')
# Pack datasets
pack_auroc_bootstrap_results(f)
pack_auwtc_bootstrap_results(f)

pack_rmst_bootstrap_results(f)
pack_shot_duration_results(f)

pack_rmst_plot_results(f)
