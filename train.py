import wandb
import yaml
import random
import numpy as np

from auton_survival.preprocessing import Preprocessor
from auton_survival.estimators import SurvivalModel

from auton_survival.metrics import survival_regression_metric

from manage_datasets import load_features_outcomes, make_training_sets, load_feature_list
from run_models import get_val_times, get_train_times


device = 'cmod'
dataset_path = 'random_flattop_256_shots_60%_disruptive'

def evaluate_model(model:SurvivalModel):
    # Make training sets if they haven't been created yet
    try:
        numeric_feats = load_feature_list(device, dataset_path)
    except:
        make_training_sets(device, dataset_path, random_seed=0)
        numeric_feats = load_feature_list(device, dataset_path)

    features_train, outcomes_train = load_features_outcomes(device, dataset_path, 'train', numeric_feats)
    features_val, outcomes_val = load_features_outcomes(device, dataset_path, 'val', numeric_feats)

        # Fit the imputer and scaler to the training data and transform the training, test, and validation data
    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
    transformer=preprocessor.fit(features_train, cat_feats=[], num_feats=numeric_feats, one_hot=True, fill_value=-1)

    x_train = transformer.transform(features_train)
    x_val = transformer.transform(features_val)

    y_train = outcomes_train
    y_val = outcomes_val

    # Get the training and validation times
    times = get_train_times(y_train)

    model.fit(x_train, y_train)

    predictions_val = model.predict_survival(x_val, times)

    metric_val = survival_regression_metric('ibs', y_val, predictions_val, times, y_train)

    return metric_val

def main():

    with open("cph_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize wandb
    run = wandb.init(project="disruption-survival-analysis", config=config)

    l2 = wandb.config.l2

    model = SurvivalModel('cph', l2=l2)

    metric_val = evaluate_model(model)

    wandb.log({'ibs': metric_val})

wandb.agent(sweep_id, function=main)