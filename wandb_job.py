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

    elif model_type == 'dsm':
        # Parameters for this type of model
        layers_str = wandb.config.layers
        layers = [int(i) for i in layers_str.split('_')]
        distribution = wandb.config.distribution
        temperature = wandb.config.temperature
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        epochs = wandb.config.epochs
        max_features = wandb.config.max_features
        k = wandb.config.k

        # Create model with parameters
        model = SurvivalModel('dsm', layers=layers, distribution=distribution, temperature=temperature, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, max_features=max_features, k=k)

metric_val = evaluate_model(device, dataset_path, model, evaluation_method, valmin, valmax, numval)

wandb.log({evaluation_method: metric_val})
