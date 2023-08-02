import os
import wandb
from model_evaluation import evaluate_model

SURVIVAL_MODELS = ['cph', 'dcph', 'dcm', 'dsm', 'rsf']

# This is a bit of a hack because I don't want to pass variables to this special 'wandb function'
device = 'cmod'
#dataset_path = 'random_flattop_256_shots_60%_disruptive'
dataset_path = 'no_ufo_flattop_1452_shots_50%_disruptive'
model_type = 'dsm'
evaluation_method = 'timeslice_ibs'

canary_file = 'wandb_job.py'

print("Running Job")

def fix_pathing():
    print("Current working directory:")
    print(os.getcwd())

    # Check if can see a file in the project directory
    print("Checking for file in project directory:")
    if os.path.isfile(canary_file):
        print("Found file")

fix_pathing()

#run = wandb.init(project='local-sweep')

print("==================")
print("======CONFIG======")
print("==================")
print(wandb.config)
print("==================")
print("======ENDFIG======")
print("==================")

model = make_model(wandb.config)

metric_val = evaluate_model(model, wandb.config)

wandb.log({evaluation_method: metric_val})
