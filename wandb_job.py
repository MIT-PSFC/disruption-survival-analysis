import os
import wandb
from model_utils import make_survival_model
from model_evaluation import evaluate_model

SURVIVAL_MODELS = ['cph', 'dcph', 'dcm', 'dsm', 'rsf']

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

if wandb.config["model_type"] in SURVIVAL_MODELS:
    model = make_survival_model(wandb.config)

#metric_val = evaluate_model(model, wandb.config)

#wandb.log({wandb.metri: metric_val})
