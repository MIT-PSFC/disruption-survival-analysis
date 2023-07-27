import sys
import wandb
from model_evaluation import evaluate_model

SURVIVAL_MODELS = ['cph', 'dcph', 'dcm', 'dsm', 'rsf']

def run_wandb():
    run = wandb.init()

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

def main():
    # This is a bit of a hack because I don't want to pass variables to this special 'wandb function'
    sweep_id = sys.argv[1]
    
    global device 
    device = sys.argv[2]

    global dataset_path
    dataset_path = sys.argv[3]

    global model_type
    model_type = sys.argv[4]

    global evaluation_method 
    evaluation_method = sys.argv[5]

    wandb.agent(sweep_id, function=run_wandb)

main()