import os
import wandb
from model_utils import make_survival_model
from model_evaluation import evaluate_model
from manage_datasets import load_features_outcomes, load_feature_list

SURVIVAL_MODELS = ['cph', 'dcph', 'dcm', 'dsm', 'rsf']
RECURRENT_MODELS = ['rdsm', 'rdcm']
FOREST_MODLES = ['rf']

run = wandb.init()

config = wandb.config

"""
print("==================")
print("======CONFIG======")
print("==================")
print(config)
print("==================")
print("======ENDFIG======")
print("==================")
"""

# Train and evaluate the model on these hyperparameters
device = config['device']
dataset_path = config['dataset_path']
numeric_feats = load_feature_list(device, dataset_path)

if config["model_type"] in SURVIVAL_MODELS:
    model = make_survival_model(config)

    x_train, y_train = load_features_outcomes(device, dataset_path, 'train', numeric_feats)
    x_val, y_val = load_features_outcomes(device, dataset_path, 'val', numeric_feats)

    try:
        model.fit(x_train, y_train)
        metric_val = evaluate_model(model, x_val, y_val, y_train, config)
    except:
        metric_val = None
else:
    metric_val = None

wandb.log({config['metric']: metric_val})
