import os
import wandb
from disruption_survival_analysis.Experiments import Experiment
# Set up WandB
os.environ["WANDB__SERVICE_WAIT"] = "800"
run = wandb.init()
config = wandb.config


# Create the experiment and try to get the evaluation metric
try:
    experiment = Experiment(config, 'val')
    metric_val = experiment.evaluate_metric(config['ab-metric'])
except:
    # If anything goes wrong during training or validation, log a None value
    metric_val = None

wandb.log({config['ab-metric']: metric_val})
