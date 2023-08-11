# Different methods of evaluating the model to have good hyperparameter tuning
import numpy as np
from auton_survival.estimators import SurvivalModel # CPH, DCPH, DSM, DCM, RSF
from auton_survival.preprocessing import Preprocessor
from auton_survival.metrics import survival_regression_metric
from manage_datasets import load_features_outcomes, load_feature_list
from Experiments import Experiment
from sklearn.ensemble import RandomForestClassifier

def evaluate_model(experiment:Experiment, config):

    # Get the metric we're evaluating the model's performance on
    metric_type = config['01_metric']

    if metric_type == 'tslic':
        # Timeslice metric. Micro avgerage over entire dataset
        # Get the validation times
        metric_val = timeslice_eval(experiment, config)
    elif metric_type == 'auroc':
        # Area under ROC curve
        metric_val = experiment.au_true_alarm_rate_false_alarm_rate_curve()
    elif metric_type == 'auwtc':
        # Area under warning time curve
        metric_val = experiment.au_warning_time_false_alarm_rate_curve()
    elif metric_type == 'maxf1':
        # Highest f1 score
        metric_val = experiment.max_f1()
    elif metric_type == 'ettdi':
        # Expected time to disruption error integral
        metric_val = experiment.ettd_diff_integral()
    else:
        metric_val = None

    return metric_val

def timeslice_eval(experiment:Experiment, config):

    # Get the model
    model = experiment.predictor.model

    # Load validation/test data
    device = config['00_device']
    dataset_path = config['00_dataset_path']

    # Load either validation or test data
    x_set, y_set = load_features_outcomes(device, dataset_path, experiment.experiment_type)

    # Validation times are hardcoded for now
    val_times = [0.1, 0.02]

    # Evaluate the model on the validation set
    try:
        if isinstance(model, SurvivalModel):
            predictions_val = model.predict_survival(x_set, val_times)
            _, y_train = load_features_outcomes(device, dataset_path, 'train')
            if np.isnan(predictions_val).any():
                return None
            else:
                metric_val = survival_regression_metric('ibs', y_set, predictions_val, val_times, y_train)
        elif isinstance(model, RandomForestClassifier):
            metric_val = model.score(x_set, y_set)
        else:
            return None
    except:
        metric_val = None

    return metric_val