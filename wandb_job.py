import sys
import wandb
from model_evaluation import evaluate_model

SURVIVAL_MODELS = ['cph', 'dcph', 'dcm', 'dsm', 'rsf']

def run_wandb():
    run = wandb.init()

    # Get the sweep name
    sweep_name = wandb.config.name

    # Split the sweep name by the '-' character
    split_name = sweep_name.split('-')
    device = split_name[0]
    dataset_path = split_name[1]
    model_type = split_name[2]
    evaluation_method = split_name[3]

    # Positions to validate the model on
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

        elif model_type == 'dcph':
            from auton_survival.estimators import SurvivalModel # CPH, DCPH, DSM, DCM, RSF


    metric_val = evaluate_model(device, dataset_path, model, evaluation_method, valmin, valmax, numval)

    wandb.log({evaluation_method: metric_val})

def main():
    sweep_id = sys.argv[1]
    wandb.agent(sweep_id, function=run_wandb)

