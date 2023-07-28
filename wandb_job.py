import sys
import wandb
from model_evaluation import evaluate_model

SURVIVAL_MODELS = ['cph', 'dcph', 'dcm', 'dsm', 'rsf']

# This is a bit of a hack because I don't want to pass variables to this special 'wandb function'
device = 'cmod'
dataset_path = 'random_flattop_256_shots_60%_disruptive'
model_type = 'cph'
evaluation_method = 'timeslice_ibs'

def run_wandb():
    run = wandb.init(project='local-sweep')

    # Positions to validate the model on
    # TODO: I want to get rid of this, just trying it out for now
    valmin = wandb.config.valmin
    valmax = wandb.config.valmax
    numval = wandb.config.numval

    if model_type in SURVIVAL_MODELS:
        from auton_survival.estimators import SurvivalModel
        
        if model_type == 'cph':
            # Parameters for this type of model
            l2 = wandb.config.l2

            # Create model with parameters
            model = SurvivalModel('cph', l2=l2)

        else:
            model = None

    metric_val = evaluate_model(device, dataset_path, model, evaluation_method, valmin, valmax, numval)

    wandb.log({evaluation_method: metric_val})

run_wandb()