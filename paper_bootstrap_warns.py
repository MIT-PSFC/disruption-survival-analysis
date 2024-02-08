from disruption_survival_analysis.sweep_config import get_bootstraps, create_bootstrap_list

from disruption_survival_analysis.experiment_utils import area_under_curve

import matplotlib.pyplot as plt
from disruption_survival_analysis.plot_experiments import pretty_name
import matplotlib as mpl

dataset_name = "paper_4"
dataset_path = f"{dataset_name}/stack_10"
all_models = ['rf', 'km', 'cph', 'dcph', 'dsm']
#all_models = ['km']

def compare_required_warning_times_auroc(device, max_warning_time):

    sets = []

    all_alarms = ['sthr']
    all_warnings = [0.01, 0.05, 0.1]

    compare_sthr_req_10ms = {'models': all_models,
                        'alarms': ['sthr'],
                        'warnings': [0.01],
                        "title": "sthr_10ms",
                        "pretty_title": "10ms AUROC",
                        "warning_string": "10ms"}
    sets.append(compare_sthr_req_10ms)

    compare_sthr_req_50ms = {'models': all_models,
                        'alarms': ['sthr'],
                        'warnings': [0.05],
                        "title": "sthr_50ms",
                        "pretty_title": "50ms AUROC",
                        "warning_string": "50ms"}
    sets.append(compare_sthr_req_50ms)

    compare_sthr_req_100ms = {'models': all_models,
                        'alarms': ['sthr'],
                        'warnings': [0.1],
                        "title": "sthr_100ms",
                        "pretty_title": "100ms AUROC",
                        "warning_string": "100ms"}
    sets.append(compare_sthr_req_100ms)


    devices = [device]
    dataset_paths = [dataset_path]
    metrics = ["auroc"]

    for set in sets:

        models = set['models']
        alarms = set['alarms']
        min_warning_times = set['warnings']
        bootstrap_list = create_bootstrap_list(devices, dataset_paths, models, alarms, metrics, min_warning_times)
        
        plt.figure()
        PLOT_STYLE = 'seaborn-v0_8-poster'
        plt.style.use(PLOT_STYLE)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.grid(True, color='w', linestyle='-', linewidth=1.5)
        plt.gca().patch.set_facecolor('0.92')
        plt.gca().set_axisbelow(True)

        #print("-------------")
        #print("ROC CURVE")
        #print("-------------")

        for bootstrapped_metrics in bootstrap_list:

            unique_fars = bootstrapped_metrics['fars']

            upper_tars = bootstrapped_metrics['upper_tars']
            lower_tars = bootstrapped_metrics['lower_tars']
            #avg_tars = bootstrapped_metrics['mean_tars']
            med_tars = bootstrapped_metrics['median_tars']

            upper_area = area_under_curve(unique_fars, upper_tars)
            lower_area = area_under_curve(unique_fars, lower_tars)
            avg_area = area_under_curve(unique_fars, med_tars)

            #print(f"{bootstrapped_metrics['name']}: {avg_area} ({lower_area}, {upper_area})")
            plt.plot(unique_fars, med_tars, label=pretty_name(bootstrapped_metrics['name']))
            plt.fill_between(unique_fars, lower_tars, upper_tars, alpha=0.3)

        plt.xlabel('FPR')
        plt.ylabel('Mean TPR')

        # put legend in lower right
        plt.legend(loc='lower right', fontsize=12)

        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # Put ticks at percents
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1],
                    ["0\%", "20\%", "40\%", "60\%", "80\%", "100\%"])
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1],
                    ["0\%", "20\%", "40\%", "60\%", "80\%", "100\%"])

        if device == 'cmod':
            plt.title(f"C-Mod {set['warning_string']} ROC Curve, Tuned for {set['pretty_title']}")
        else:
            plt.title(f"DIII-D {set['warning_string']} ROC Curve, Tuned for {set['pretty_title']}")
        plt.savefig(f"plots/{dataset_name}/auroc/{devices[0]}_{set['title']}_roc.png", bbox_inches='tight')

        plt.rcParams.update(mpl.rcParamsDefault)
        plt.figure()

        plt.style.use(PLOT_STYLE)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.grid(True, color='w', linestyle='-', linewidth=1.5)
        plt.gca().patch.set_facecolor('0.92')
        plt.gca().set_axisbelow(True)

        #print("-------------")
        #print("WARNING TIMES")
        #print("-------------")

        for bootstrapped_metrics in bootstrap_list:

            unique_fars = bootstrapped_metrics['fars']

            upper_warns = bootstrapped_metrics['upper_warns']
            lower_warns = bootstrapped_metrics['lower_warns']

            upper_area = area_under_curve(unique_fars, upper_warns, x_cutoff=0.05)
            lower_area = area_under_curve(unique_fars, lower_warns, x_cutoff=0.05)
            med_area = area_under_curve(unique_fars, bootstrapped_metrics['median_warns'], x_cutoff=0.05)

            #print(f"{bootstrapped_metrics['name']}: {avg_area*1000} ({lower_area*1000}, {upper_area*1000})")
            
            plt.plot(unique_fars, bootstrapped_metrics['median_warns']*1000, label=pretty_name(bootstrapped_metrics['name']))
            plt.fill_between(unique_fars, lower_warns*1000, upper_warns*1000, alpha=0.3)

        plt.xlabel('FPR')
        plt.ylabel('IQM Warning Time [ms]')


        plt.xlim([0, 0.05])
        plt.ylim([0, max_warning_time])

        plt.xscale('symlog')

        # Make x ticks go from 0 to 0.05 in increments of 0.01
        plt.xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05],
                    ["0\%", "1\%", "2\%", "3\%", "4\%", "5\%"])

        plt.legend(loc='upper left', fontsize=12)

        if device == 'cmod':
            plt.title(f"C-Mod WTC Curve, Tuned for {set['pretty_title']}")
        else:
            plt.title(f"DIII-D WTC Curve, Tuned for {set['pretty_title']}")
        plt.savefig(f"plots/{dataset_name}/auroc/{devices[0]}_{set['title']}_warning_times.png", bbox_inches='tight')
        plt.rcParams.update(mpl.rcParamsDefault)

def compare_required_warning_times_auwtc(device, max_warning_time):

    sets = []

    all_alarms = ['sthr']
    all_warnings = [0.01, 0.05, 0.1]

    compare_sthr_req_10ms = {'models': all_models,
                        'alarms': ['sthr'],
                        'warnings': [0.01],
                        "title": "sthr_10ms",
                        "pretty_title": "AUWTC",
                        "warning_string": "10ms"}
    sets.append(compare_sthr_req_10ms)

    compare_sthr_req_50ms = {'models': all_models,
                        'alarms': ['sthr'],
                        'warnings': [0.05],
                        "title": "sthr_50ms",
                        "pretty_title": "AUWTC",
                        "warning_string": "50ms"}
    sets.append(compare_sthr_req_50ms)

    compare_sthr_req_100ms = {'models': all_models,
                        'alarms': ['sthr'],
                        'warnings': [0.1],
                        "title": "sthr_100ms",
                        "pretty_title": "AUWTC",
                        "warning_string": "100ms"}
    sets.append(compare_sthr_req_100ms)

    devices = [device]
    dataset_paths = [dataset_path]
    metrics = ["auwtc"]

    for set in sets:

        models = set['models']
        alarms = set['alarms']
        min_warning_times = set['warnings']
        bootstrap_list = create_bootstrap_list(devices, dataset_paths, models, alarms, metrics, min_warning_times)
        
        plt.figure()
        PLOT_STYLE = 'seaborn-v0_8-poster'
        plt.style.use(PLOT_STYLE)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.grid(True, color='w', linestyle='-', linewidth=1.5)
        plt.gca().patch.set_facecolor('0.92')
        plt.gca().set_axisbelow(True)

        #print("-------------")
        #print("ROC CURVE")
        #print("-------------")

        for bootstrapped_metrics in bootstrap_list:

            unique_fars = bootstrapped_metrics['fars']

            upper_tars = bootstrapped_metrics['upper_tars']
            lower_tars = bootstrapped_metrics['lower_tars']
            #avg_tars = bootstrapped_metrics['mean_tars']
            med_tars = bootstrapped_metrics['median_tars']

            upper_area = area_under_curve(unique_fars, upper_tars)
            lower_area = area_under_curve(unique_fars, lower_tars)
            avg_area = area_under_curve(unique_fars, med_tars)

            #print(f"{bootstrapped_metrics['name']}: {avg_area} ({lower_area}, {upper_area})")
            plt.plot(unique_fars, med_tars, label=pretty_name(bootstrapped_metrics['name']))
            plt.fill_between(unique_fars, lower_tars, upper_tars, alpha=0.3)

        plt.xlabel('FPR')
        plt.ylabel('Mean TPR')

        # put legend in lower right
        plt.legend(loc='lower right', fontsize=12)

        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # Put ticks at percents
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1],
                    ["0\%", "20\%", "40\%", "60\%", "80\%", "100\%"])
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1],
                    ["0\%", "20\%", "40\%", "60\%", "80\%", "100\%"])

        if device == 'cmod':
            plt.title(f"C-Mod {set['warning_string']} ROC Curve, Tuned for {set['pretty_title']}")
        else:
            plt.title(f"DIII-D {set['warning_string']} ROC Curve, Tuned for {set['pretty_title']}")
        
        plt.savefig(f"plots/{dataset_name}/auwtc/{devices[0]}_{set['title']}_roc.png", bbox_inches='tight')

        plt.rcParams.update(mpl.rcParamsDefault)
        plt.figure()

        plt.style.use(PLOT_STYLE)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.grid(True, color='w', linestyle='-', linewidth=1.5)
        plt.gca().patch.set_facecolor('0.92')
        plt.gca().set_axisbelow(True)

        #print("-------------")
        #print("WARNING TIMES")
        #print("-------------")

        for bootstrapped_metrics in bootstrap_list:

            unique_fars = bootstrapped_metrics['fars']

            upper_warns = bootstrapped_metrics['upper_warns']
            lower_warns = bootstrapped_metrics['lower_warns']

            upper_area = area_under_curve(unique_fars, upper_warns, x_cutoff=0.05)
            lower_area = area_under_curve(unique_fars, lower_warns, x_cutoff=0.05)
            med_area = area_under_curve(unique_fars, bootstrapped_metrics['median_warns'], x_cutoff=0.05)

            #print(f"{bootstrapped_metrics['name']}: {avg_area*1000} ({lower_area*1000}, {upper_area*1000})")

            plt.plot(unique_fars, bootstrapped_metrics['median_warns']*1000, label=pretty_name(bootstrapped_metrics['name']))
            plt.fill_between(unique_fars, lower_warns*1000, upper_warns*1000, alpha=0.3)

        plt.xlabel('FPR')
        plt.ylabel('IQM Warning Time [ms]')


        plt.xlim([0, 0.05])
        plt.ylim([0, max_warning_time])

        plt.xscale('symlog')

        # Make x ticks go from 0 to 0.05 in increments of 0.01
        plt.xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05],
                    ["0\%", "1\%", "2\%", "3\%", "4\%", "5\%"])

        plt.legend(loc='upper left', fontsize=12)

        if device == 'cmod':
            plt.title(f"C-Mod WTC Curve, Tuned for {set['pretty_title']}")
        else:
            plt.title(f"DIII-D WTC Curve, Tuned for {set['pretty_title']}")
        plt.savefig(f"plots/{dataset_name}/auwtc/{devices[0]}_{set['title']}_warning_times.png", bbox_inches='tight')
        plt.rcParams.update(mpl.rcParamsDefault)


def compare_required_warning_times_auroc_FNR(device, max_warning_time):

    sets = []

    all_alarms = ['sthr']
    all_warnings = [0.01, 0.05, 0.1]

    compare_sthr_req_10ms = {'models': all_models,
                        'alarms': ['sthr'],
                        'warnings': [0.01],
                        "title": "sthr_10ms",
                        "pretty_title": "10ms AUROC",
                        "warning_string": "10ms"}
    sets.append(compare_sthr_req_10ms)

    compare_sthr_req_50ms = {'models': all_models,
                        'alarms': ['sthr'],
                        'warnings': [0.05],
                        "title": "sthr_50ms",
                        "pretty_title": "50ms AUROC",
                        "warning_string": "50ms"}
    sets.append(compare_sthr_req_50ms)

    compare_sthr_req_100ms = {'models': all_models,
                        'alarms': ['sthr'],
                        'warnings': [0.1],
                        "title": "sthr_100ms",
                        "pretty_title": "100ms AUROC",
                        "warning_string": "100ms"}
    sets.append(compare_sthr_req_100ms)


    devices = [device]
    dataset_paths = [dataset_path]
    metrics = ["auroc"]

    for set in sets:

        models = set['models']
        alarms = set['alarms']
        min_warning_times = set['warnings']
        bootstrap_list = create_bootstrap_list(devices, dataset_paths, models, alarms, metrics, min_warning_times)
        
        plt.figure()
        PLOT_STYLE = 'seaborn-v0_8-poster'
        plt.style.use(PLOT_STYLE)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.grid(True, color='w', linestyle='-', linewidth=1.5)
        plt.gca().patch.set_facecolor('0.92')
        plt.gca().set_axisbelow(True)

        #print("-------------")
        #print("WARNING TIMES")
        #print("-------------")

        for bootstrapped_metrics in bootstrap_list:

            unique_fars = bootstrapped_metrics['fars']

            upper_warns = bootstrapped_metrics['upper_warns']
            lower_warns = bootstrapped_metrics['lower_warns']

            upper_area = area_under_curve(unique_fars, upper_warns, x_cutoff=0.05)
            lower_area = area_under_curve(unique_fars, lower_warns, x_cutoff=0.05)
            med_area = area_under_curve(unique_fars, bootstrapped_metrics['median_warns'], x_cutoff=0.05)

            #print(f"{bootstrapped_metrics['name']}: {avg_area*1000} ({lower_area*1000}, {upper_area*1000})")
            
            plt.plot(1-unique_fars, bootstrapped_metrics['median_warns']*1000, label=pretty_name(bootstrapped_metrics['name']))
            plt.fill_between(1-unique_fars, lower_warns*1000, upper_warns*1000, alpha=0.3)

        plt.xlabel('FNR')
        plt.ylabel('IQM Warning Time [ms]')

        plt.ylim([0, max_warning_time])
        plt.xlim([0, 1])

        # Make x ticks go from 0 to 1 in increments of 0.2
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1],
                    ["0\%", "20\%", "40\%", "60\%", "80\%", "100\%"])

        plt.legend(loc='lower left', fontsize=12)

        if device == 'cmod':
            plt.title(f"C-Mod FNR Curve, Tuned for {set['pretty_title']}")
        else:
            plt.title(f"DIII-D FNR Curve, Tuned for {set['pretty_title']}")
        plt.savefig(f"plots/{devices[0]}_{set['title']}_warning_times_FNR.png", bbox_inches='tight')
        plt.rcParams.update(mpl.rcParamsDefault)