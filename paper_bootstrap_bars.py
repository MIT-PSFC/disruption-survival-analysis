from disruption_survival_analysis.sweep_config import get_bootstraps, create_bootstrap_list

from disruption_survival_analysis.experiment_utils import area_under_curve

import matplotlib.pyplot as plt
from disruption_survival_analysis.plot_experiments import pretty_name
import matplotlib as mpl

import numpy as np

dataset_name = "paper_4"
dataset_path = f"{dataset_name}/stack_10"
all_models = ['rf', 'km', 'cph', 'dcph', 'dsm']

def compare_required_warning_times_bars(device, tuned_metric, plot_metric, title, max_area=1):

    sets = []

    all_alarms = ['sthr']
    all_warnings = [0.01, 0.05, 0.1]
    warn_colors = ['red', 'orange', 'green']

    for model in all_models:
        sets.append({'models': [model],
                     'alarms': all_alarms,
                     'warnings': all_warnings,
                     "title": f"{model}"})

    devices = [device]
    dataset_paths = [dataset_path]
    metrics = [tuned_metric]

    plt.figure()
    PLOT_STYLE = 'seaborn-v0_8-poster'
    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.grid(True, color='w', linestyle='-', linewidth=1.5)
    plt.gca().patch.set_facecolor('0.92')
    plt.gca().set_axisbelow(True)

    x = np.arange(len(sets))
    width = 0.2

    for i, set in enumerate(sets):
        models = set['models']
        alarms = set['alarms']
        min_warning_times = set['warnings']
        bootstrap_list = create_bootstrap_list(devices, dataset_paths, models, alarms, metrics, min_warning_times)

        for j, bootstrapped_metrics in enumerate(bootstrap_list):
            unique_fars = bootstrapped_metrics['fars']
            offset = (j-1)*width

            if plot_metric == 'auroc':
                upper_tars = bootstrapped_metrics['upper_tars']
                lower_tars = bootstrapped_metrics['lower_tars']
                #avg_tars = bootstrapped_metrics['mean_tars']
                med_tars = bootstrapped_metrics['median_tars']
                upper_area = area_under_curve(unique_fars, upper_tars)
                lower_area = area_under_curve(unique_fars, lower_tars)
                typ_area = area_under_curve(unique_fars, med_tars)
            elif plot_metric == 'auwtc':
                upper_warns = bootstrapped_metrics['upper_warns']
                lower_warns = bootstrapped_metrics['lower_warns']
                upper_area = area_under_curve(unique_fars, upper_warns, x_cutoff=0.05)*1000
                lower_area = area_under_curve(unique_fars, lower_warns, x_cutoff=0.05)*1000
                typ_area = area_under_curve(unique_fars, bootstrapped_metrics['median_warns'], x_cutoff=0.05)*1000
            else:
                raise Exception(f"Unknown plot metric: {plot_metric}")

            if i == 0:
                    plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"{min_warning_times[j]*1000} ms")
            else:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j])
            upper_error_bar = max(0, upper_area - typ_area)
            lower_error_bar = max(0, typ_area - lower_area)
            plt.errorbar(x[i]+offset, typ_area, yerr=[[upper_error_bar], [lower_error_bar]], fmt='', ecolor='k', capsize=10)

    plt.xticks(x, [set['title'].upper() for set in sets])
    plt.ylabel(f"{plot_metric.upper()}")
    #plt.xlabel("MODEL")
    plt.ylim([0, max_area])

    #plt.legend(loc='upper left', fontsize=12)

    plt.title(title)
    plt.savefig(f"plots/{dataset_name}/{tuned_metric}/{devices[0]}_{plot_metric}_bar.png", bbox_inches='tight')

    plt.rcParams.update(mpl.rcParamsDefault)
    plt.figure()
