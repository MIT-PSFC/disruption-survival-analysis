from paper_bootstrap_warns import compare_required_warning_times_auroc, compare_required_warning_times_auwtc, compare_required_warning_times_auroc_FNR

from paper_bootstrap_bars import compare_required_warning_times_bars

from paper_bootstrap_combined import compare_auwtc_results, compare_auroc_results, compare_rmst_results

from paper_bootstrap_combined import compare_auwtc_results_rmst, compare_auroc_results_rmst

#compare_required_warning_times_auroc("cmod", 60)
#compare_required_warning_times_auwtc("cmod", 60)
#compare_required_warning_times_auroc("d3d", 700)
#compare_required_warning_times_auwtc("d3d", 700)
#compare_required_warning_times_auroc_FNR("d3d", 3000)

# compare_required_warning_times_bars("cmod", "auroc", "auroc", "C-Mod AUROC, Tuned for AUROC")
# compare_required_warning_times_bars("cmod", "auroc", "auwtc", "C-Mod AUWTC, Tuned for AUROC", max_area=2)
# compare_required_warning_times_bars("cmod", "auwtc", "auroc", "C-Mod, AUROC, Tuned for AUWTC")
# compare_required_warning_times_bars("cmod", "auwtc", "auwtc", "C-Mod, AUWTC, Tuned for AUWTC", max_area=2)

# compare_required_warning_times_bars("d3d", "auroc", "auroc", "DIII-D AUROC, Tuned for AUROC")
# compare_required_warning_times_bars("d3d", "auroc", "auwtc", "DIII-D AUWTC, Tuned for AUROC", max_area=16)
# compare_required_warning_times_bars("d3d", "auwtc", "auroc", "DIII-D AUROC, Tuned for AUWTC")
# compare_required_warning_times_bars("d3d", "auwtc", "auwtc", "DIII-D AUWTC, Tuned for AUWTC", max_area=16)

compare_auwtc_results("cmod", "C-Mod AUWTC", 1.75)
compare_auwtc_results("d3d", "DIII-D AUWTC", 16)

compare_auroc_results("cmod", "C-Mod AUROC")
compare_auroc_results("d3d", "DIII-D AUROC")

# compare_auwtc_results_rmst("cmod", "C-Mod AUWTC", 1.75)
# compare_auwtc_results_rmst("d3d", "DIII-D AUWTC", 16)

# compare_auroc_results_rmst("cmod", "C-Mod AUROC")
# compare_auroc_results_rmst("d3d", "DIII-D AUROC")

# #compare_rmst_results"cmod", 'all', "C-Mod RMST", 1)
# #compare_rmst_results("d3d", 'all', "DIII-D RMST", 1)
# compare_rmst_results("cmod", 'disruptive', "C-Mod Disruptive RMST Difference", 0.35)
# compare_rmst_results("d3d", 'disruptive', "DIII-D Disruptive RMST Difference", 0.35)
# compare_rmst_results("cmod", 'non_disruptive', "C-Mod Non-Disruptive RMST Difference", 0.06)
# compare_rmst_results("d3d", 'non_disruptive', "DIII-D Non-Disruptive RMST Difference", 0.06)(