# Different methods of evaluating the model to have good hyperparameter tuning
import numpy as np
from auton_survival.estimators import SurvivalModel # CPH, DCPH, DSM, DCM, RSF
from auton_survival.preprocessing import Preprocessor
from auton_survival.metrics import survival_regression_metric
from manage_datasets import load_features_outcomes, load_feature_list

def get_val_times(y_tr, min_quantile, max_quantile):
    """
    Get the validation times for survival models
    """
    #return np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(min_quantile, max_quantile, 10)).tolist()
    return [0.02, 0.1] # The two most important threshold times according to Ryan

def evaluate_model(model, config):

    # Get the metric we're evaluating the model's performance on
    metric_type = config['metric']
    numeric_feats = load_feature_list(device, dataset_path)

    if metric_type == 'timeslice_ibs':
        # Only applicable to survival models
        x_val, y_val = load_features_outcomes(device, dataset_path, 'val', numeric_feats)
        val_times = get_val_times(y_train, config['valmin'], config['valmax'])
        metric_val = timeslice_eval(model, x_val, y_val, y_train, val_times)
    elif metric_type == 'timeslice_score':
        # Only applicable to binary classifiers
        x_val, y_val = load_features_outcomes(device, dataset_path, 'val', numeric_feats)
        metric_val = model.score(x_val, y_val)
    elif metric_type == 'au_roc_simple_threshold':
        #experiment = 

        metric_val = experiment.
    else:
        metric_val = None

    return metric_val

def timeslice_eval(model, x_val, y_val, y_train, val_times):

    # Evaluate the model on the validation set
    try:
        if isinstance(model, SurvivalModel):
            predictions_val = model.predict_survival(x_val, val_times)

            if np.isnan(predictions_val).any():
                return None
            else:
                metric_val = survival_regression_metric('ibs', y_val, predictions_val, val_times, y_train)
        else:
            return None
    except:
        metric_val = None

    return metric_val

def au_roc_simple_threshold(model, ):



#    return

