import os
import wandb
from model_utils import make_survival_model
from model_evaluation import evaluate_model

SURVIVAL_MODELS = ['cph', 'dcph', 'dcm', 'dsm', 'rsf']

run = wandb.init()

"""
print("==================")
print("======CONFIG======")
print("==================")
print(wandb.config)
print("==================")
print("======ENDFIG======")
print("==================")
"""

if wandb.config["model_type"] in SURVIVAL_MODELS:
    model = make_survival_model(wandb.config)

#metric_val = evaluate_model(model, wandb.config)

#wandb.log({wandb.metri: metric_val})
