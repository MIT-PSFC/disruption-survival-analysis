from disruption_survival_analysis.manage_datasets import make_training_sets, make_stacked_sets

device = 'cmod'
dataset_path = 'sql_match'

#device = 'synthetic'
#dataset_path = 'test'


# Make training sets if they haven't been created yet
make_training_sets(device, dataset_path, random_seed=0)

# Make temporal datasets
stack_sizes = [10]
for stack_size in stack_sizes:
    make_stacked_sets(device, dataset_path, 'train', stack_size)
    make_stacked_sets(device, dataset_path, 'test', stack_size)
    make_stacked_sets(device, dataset_path, 'val', stack_size)