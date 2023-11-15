from disruption_survival_analysis.manage_datasets import make_training_sets, make_stacked_sets, focus_training_set

#device = 'd3d'
#dataset_path = 'matlab_flattop'
device = 'synthetic'
dataset_path = 'small'


# Make training sets if they haven't been created yet
make_training_sets(device, dataset_path, random_seed=0)
# focus_training_set(device, dataset_path)

# Make temporal datasets
stack_sizes = [10]
for stack_size in stack_sizes:
    make_stacked_sets(device, dataset_path, 'train_full', stack_size)
    make_stacked_sets(device, dataset_path, 'test', stack_size)
    make_stacked_sets(device, dataset_path, 'val', stack_size)
    focus_training_set(device, f"{dataset_path}/stack_{stack_size}")

#focus_training_set(device, f"{dataset_path}/stack_5")
