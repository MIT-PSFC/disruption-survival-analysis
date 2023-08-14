import os
import wandb
from Experiments import make_experiment
from model_evaluation import evaluate_model

# Set up WandB
os.environ["WANDB__SERVICE_WAIT"] = "800"
run = wandb.init()
config = wandb.config


# Create the experiment and try to get the evaluation metric
try:
    experiment = make_experiment(config, 'val')
    metric_val = evaluate_model(experiment, config)
except:
    # If anything goes wrong during training or validation, log a None value
    metric_val = None

wandb.log({config['ab-metric']: metric_val})
