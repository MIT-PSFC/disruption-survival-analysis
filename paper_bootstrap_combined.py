from disruption_survival_analysis.sweep_config import create_bootstrap_list, create_rmst_list

from disruption_survival_analysis.experiment_utils import area_under_curve

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

dataset_name = "paper_4"
dataset_path = f"{dataset_name}/stack_10"
all_models = ['rf', 'km', 'cph', 'dcph', 'dsm']

def compare_auwtc_results(device, title, max_area = 1.0):

    sets = []
    all_alarms = ['sthr']
    all_warnings = [0.01, 0.05, 0.1]
    warn_colors = ['red', 'orange', 'yellow', 'blue']

    for model in all_models:
        sets.append({'models': [model],
                     'alarms': all_alarms,
                     'warnings': all_warnings,
                     "title": f"{model}"})
    
    plt.figure()
    PLOT_STYLE = 'seaborn-v0_8-poster'
    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.grid(True, color='w', linestyle='-', linewidth=1.5)
    plt.gca().patch.set_facecolor('0.92')
    plt.gca().set_axisbelow(True)

    x = np.arange(len(sets))
    width = 0.18

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
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"ROC {min_warning_times[j]*1000} ms")
            elif i == 0 and j == 3:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"WTC All")
            else:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j])
            upper_error_bar = max(0, upper_area - typ_area)
            lower_error_bar = max(0, typ_area - lower_area)
            plt.errorbar(x[i]+offset, typ_area, yerr=[[lower_error_bar], [upper_error_bar]], fmt='', ecolor='k', capsize=10)

    if device == 'cmod':
        plt.legend(loc='upper right', fontsize=16)
    else:
        plt.legend(loc='upper left', fontsize=16)

    plt.xticks(x, [set['title'].upper() for set in sets])
    plt.ylabel(f"Bootstrap AUWTC [ms]")
    #plt.xlabel("MODEL")
    plt.ylim([0, max_area])

    plt.title(title)
    plt.savefig(f"plots/{dataset_name}/combined/{device}_auwtc_bar.png", bbox_inches='tight')

    plt.rcParams.update(mpl.rcParamsDefault)
    plt.figure()

def compare_auroc_results(device, title, max_area = 1):

    sets = []
    all_alarms = ['sthr']
    all_warnings = [0.01, 0.05, 0.1]
    warn_colors = ['red', 'orange', 'yellow', 'blue', 'purple', 'cyan']

    for model in all_models:
        sets.append({'models': [model],
                     'alarms': all_alarms,
                     'warnings': all_warnings,
                     "title": f"{model}"})
    
    plt.figure()
    PLOT_STYLE = 'seaborn-v0_8-poster'
    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.grid(True, color='w', linestyle='-', linewidth=1.5)
    plt.gca().patch.set_facecolor('0.92')
    plt.gca().set_axisbelow(True)

    x = np.arange(len(sets))
    width = 0.12

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
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"ROC {min_warning_times[j]*1000} ms")
            elif i == 0 and j >= 3:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"WTC {min_warning_times[j-3]*1000} ms")
            else:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j])
            upper_error_bar = max(0, upper_area - typ_area)
            lower_error_bar = max(0, typ_area - lower_area)
            plt.errorbar(x[i]+offset, typ_area, yerr=[[lower_error_bar], [upper_error_bar]], fmt='', ecolor='k', capsize=10)

    if device == 'cmod':
        plt.legend(loc='upper left', fontsize=13, ncol=2)
    else:
        #plt.legend(loc='upper left', fontsize=16)
        pass

    plt.xticks(x, [set['title'].upper() for set in sets])
    plt.ylabel(f"Bootstrap AUROC")
    #plt.xlabel("MODEL")
    plt.ylim([0.4, max_area])

    plt.title(title)
    plt.savefig(f"plots/{dataset_name}/combined/{device}_auroc_bar.png", bbox_inches='tight')

    plt.rcParams.update(mpl.rcParamsDefault)
    plt.figure()

def compare_auwtc_results_rmst(device, title, max_area = 1.0):

    sets = []
    all_alarms = ['sthr']
    all_warnings = [0.01, 0.05, 0.1]
    warn_colors = ['red', 'orange', 'yellow', 'blue', 'green']

    for model in all_models:
        sets.append({'models': [model],
                     'alarms': all_alarms,
                     'warnings': all_warnings,
                     "title": f"{model}"})
    
    plt.figure()
    PLOT_STYLE = 'seaborn-v0_8-poster'
    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.grid(True, color='w', linestyle='-', linewidth=1.5)
    plt.gca().patch.set_facecolor('0.92')
    plt.gca().set_axisbelow(True)

    x = np.arange(len(sets))
    width = 0.18

    for i, set in enumerate(sets):
        models = set['models']
        alarms = set['alarms']
        min_warning_times = set['warnings']
        full_bootstrap_list = create_bootstrap_list([device], [dataset_path], models, alarms, ["auroc", "auwtc", "rmstid"], min_warning_times)

        # Only keep 50ms AUWTC and RMSTID
        bootstrap_list = []
        for k, item in enumerate(full_bootstrap_list):
             if k in [0, 1, 2, 4, 6]:
                bootstrap_list.append(item)

        for j, bootstrapped_metrics in enumerate(bootstrap_list):
            unique_fars = bootstrapped_metrics['fars']
            offset = (j-2)*width

            upper_warns = bootstrapped_metrics['upper_warns']
            lower_warns = bootstrapped_metrics['lower_warns']
            upper_area = area_under_curve(unique_fars, upper_warns, x_cutoff=0.05)*1000
            lower_area = area_under_curve(unique_fars, lower_warns, x_cutoff=0.05)*1000
            typ_area = area_under_curve(unique_fars, bootstrapped_metrics['median_warns'], x_cutoff=0.05)*1000

            if i == 0 and j < 3:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"ROC {min_warning_times[j]*1000} ms")
            elif i == 0 and j == 3:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"WTC All")
            elif i == 0 and j == 4:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"RMST All")
            else:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j])
            upper_error_bar = max(0, upper_area - typ_area)
            lower_error_bar = max(0, typ_area - lower_area)
            plt.errorbar(x[i]+offset, typ_area, yerr=[[lower_error_bar], [upper_error_bar]], fmt='', ecolor='k', capsize=10)

    if device == 'cmod':
        plt.legend(loc='upper right', fontsize=16)
    else:
        plt.legend(loc='upper left', fontsize=16)

    plt.xticks(x, [set['title'].upper() for set in sets])
    plt.ylabel(f"Bootstrap AUWTC [ms]")
    #plt.xlabel("MODEL")
    plt.ylim([0, max_area])

    plt.title(title)
    plt.savefig(f"plots/{dataset_name}/combined/{device}_auwtc_bar_rmst.png", bbox_inches='tight')

    plt.rcParams.update(mpl.rcParamsDefault)
    plt.figure()

def compare_auroc_results_rmst(device, title, max_area = 1):

    sets = []
    all_alarms = ['sthr']
    all_warnings = [0.01, 0.05, 0.1]
    warn_colors = ['red', 'orange', 'yellow', 'blue', 'purple', 'cyan', 'green', 'olive', 'brown']

    for model in all_models:
        sets.append({'models': [model],
                     'alarms': all_alarms,
                     'warnings': all_warnings,
                     "title": f"{model}"})
    
    plt.figure()
    PLOT_STYLE = 'seaborn-v0_8-poster'
    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.grid(True, color='w', linestyle='-', linewidth=1.5)
    plt.gca().patch.set_facecolor('0.92')
    plt.gca().set_axisbelow(True)

    x = np.arange(len(sets))
    width = 0.09

    for i, set in enumerate(sets):
        models = set['models']
        alarms = set['alarms']
        min_warning_times = set['warnings']
        bootstrap_list = create_bootstrap_list([device], [dataset_path], models, alarms, ["auroc", "auwtc", "rmstid"], min_warning_times)

        for j, bootstrapped_metrics in enumerate(bootstrap_list):
            unique_fars = bootstrapped_metrics['fars']
            offset = (j-4)*width

            upper_tars = bootstrapped_metrics['upper_tars']
            lower_tars = bootstrapped_metrics['lower_tars']
            #avg_tars = bootstrapped_metrics['mean_tars']
            med_tars = bootstrapped_metrics['median_tars']
            upper_area = area_under_curve(unique_fars, upper_tars)
            lower_area = area_under_curve(unique_fars, lower_tars)
            typ_area = area_under_curve(unique_fars, med_tars)

            if i == 0 and j < 3:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"ROC {min_warning_times[j]*1000} ms")
            elif i == 0 and j >= 3 and j < 6:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"WTC {min_warning_times[j-3]*1000} ms")
            elif i == 0 and j >= 6:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j], label=f"RMST {min_warning_times[j-6]*1000} ms")
            else:
                plt.bar(x[i]+offset, typ_area, width, color=warn_colors[j])
            upper_error_bar = max(0, upper_area - typ_area)
            lower_error_bar = max(0, typ_area - lower_area)
            plt.errorbar(x[i]+offset, typ_area, yerr=[[lower_error_bar], [upper_error_bar]], fmt='', ecolor='k', capsize=10)

    if device == 'cmod':
        plt.legend(loc='upper left', fontsize=10, ncol=3)
    else:
        #plt.legend(loc='upper left', fontsize=16)
        pass

    plt.xticks(x, [set['title'].upper() for set in sets])
    plt.ylabel(f"Bootstrap AUROC")
    #plt.xlabel("MODEL")
    plt.ylim([0, max_area])

    plt.title(title)
    plt.savefig(f"plots/{dataset_name}/combined/{device}_auroc_bar_rmst.png", bbox_inches='tight')

    plt.rcParams.update(mpl.rcParamsDefault)
    plt.figure()

def compare_rmst_results(device, shot_group, title, max_display=1.0):
    
    plt.figure()
    PLOT_STYLE = 'seaborn-v0_8-poster'
    plt.style.use(PLOT_STYLE)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.grid(True, color='w', linestyle='-', linewidth=1.5)
    plt.gca().patch.set_facecolor('0.92')
    plt.gca().set_axisbelow(True)

    sets = []
    all_alarms = ['sthr']
    all_warnings = [0.01, 0.05, 0.1]
    warn_colors = ['red', 'orange', 'yellow', 'blue', 'green']

    for model in all_models:
        sets.append({'models': [model],
                     'alarms': all_alarms,
                     'warnings': all_warnings,
                     "title": f"{model}"})
        
    x = np.arange(len(sets))
    width = 0.18

    for i, set in enumerate(sets):
        models = set['models']
        alarms = set['alarms']
        min_warning_times = set['warnings']
        full_rmst_list = create_rmst_list([device], [dataset_path], models, alarms, ["auroc", "auwtc", "rmstsl"], min_warning_times)

        # Only keep 10ms AUWTC and RMSTID (same model, gives same RMST)
        rmst_list = []
        for k, item in enumerate(full_rmst_list):
             if k in [0, 1, 2, 3, 4]:
                rmst_list.append(item)

        for j, bootstrapped_metrics in enumerate(rmst_list):
            if shot_group == 'disruptive':
                rmst_results = bootstrapped_metrics['disruptive_results']
            elif shot_group == 'non_disruptive':
                rmst_results = bootstrapped_metrics['non_disruptive_results']
            elif shot_group == 'all':
                rmst_results = bootstrapped_metrics['all_results']
            else:
                raise Exception(f"Unknown shot group: {shot_group}")
            
            offset = (j-1.5)*width

            upper_rmst = rmst_results['iq3']
            lower_rmst = rmst_results['iq1']
            typ_rmst = rmst_results['med']

            if i == 0 and j < 3:
                plt.bar(x[i]+offset, typ_rmst, width, color=warn_colors[j], label=f"ROC {min_warning_times[j]*1000} ms")
            elif i == 0 and j == 3:
                plt.bar(x[i]+offset, typ_rmst, width, color=warn_colors[j], label=f"WTC All")
            elif i == 0 and j == 4:
                plt.bar(x[i]+offset, typ_rmst, width, color=warn_colors[j], label=f"RMST All")
            else:
                plt.bar(x[i]+offset, typ_rmst, width, color=warn_colors[j])
            upper_error_bar = max(0, upper_rmst - typ_rmst)
            lower_error_bar = max(0, typ_rmst - lower_rmst)
            plt.errorbar(x[i]+offset, typ_rmst, yerr=[[lower_error_bar], [upper_error_bar]], fmt='', ecolor='k', capsize=10)

    if device == 'cmod':
        plt.legend(loc='upper left', fontsize=13, ncol=2)
    else:
        #plt.legend(loc='upper left', fontsize=16)
        pass

    plt.xticks(x, [set['title'].upper() for set in sets])
    plt.ylabel(f"RMST Difference Integral [{r'$s^2$'}]")

    plt.ylim([0, max_display])
    
    plt.title(title)
    plt.savefig(f"plots/{dataset_name}/combined/{device}_rmst_bar_{shot_group}.png", bbox_inches='tight')
        
    plt.rcParams.update(mpl.rcParamsDefault)
    plt.figure()