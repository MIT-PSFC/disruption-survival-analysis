import os
import wandb
from model_utils import make_model
from model_evaluation import evaluate_model
from manage_datasets import load_features_outcomes, load_features_labels, load_feature_list

# Different model types must be trained and scored slightly differently
SURVIVAL_MODELS = ['cph', 'dcph', 'dcm', 'dsm', 'rsf']
RECURRENT_SURVIVAL_MODELS = ['rdsm', 'rdcm']
BINARY_CLASSIFIERS = ['rf']

# Set up WandB
os.environ["WANDB__SERVICE_WAIT"] = "800"
run = wandb.init()
config = wandb.config

# Get training dataset path from config
device = config['device']
dataset_path = config['dataset_path']
numeric_feats = load_feature_list(device, dataset_path)

# Create model from config
model = make_model(config)

# Train and evaluate model based on model type
if config["model_type"] in SURVIVAL_MODELS:
    # Load training sets
    x_train, y_train = load_features_outcomes(device, dataset_path, 'train', numeric_feats)
    
    # Try to train and evaluate the model with this hyperparameter config
    # If anything goes wrong, return metric val of 'none'
    try:
        model.fit(x_train, y_train)
        metric_val = evaluate_model(model, device, dataset_path, y_train, config)
    except:
        metric_val = None

elif config["model_type"] in BINARY_CLASSIFIERS:
    # Load training and validation sets
    # Binary classification disruptive window labeling is a hyperparameter
    disruptive_window = config['disruptive_window']
    x_train, labels_train = load_features_labels(device, dataset_path, 'train', disruptive_window, numeric_feats)
    x_val, labels_val = load_features_labels(device, dataset_path, 'val', disruptive_window, numeric_feats)

    # Try to train and evaluate the model with this hyperparameter config
    # If anything goes wrong, return metric val of 'none'
    try:
        model.fit(x_train, labels_train)
        metric_val = evaluate_model(model, x_val, labels_val, labels_train, config)
    except:
        metric_val = None

else:
    metric_val = None

wandb.log({config['metric']: metric_val})
