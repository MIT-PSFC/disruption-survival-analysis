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
    return np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(min_quantile, max_quantile, 10)).tolist()

def evaluate_model(model, x_val, y_val, y_train, config):

    # Get the metric we're evaluating the model's performance on
    metric_type = config['metric']

    if metric_type == 'timeslice_ibs':
        val_times = get_val_times(y_train, config['valmin'], config['valmax'])
        metric_val = timeslice_eval(model, x_val, y_val, y_train, val_times)
    elif metric_type == 'missed_alarm':
        #metric_val = missed_alarm_eval(model, device, dataset_path)
        metric_val = None
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

#def missed_alarm_eval():


#    return

def disruptive_shot_expected_lifetime_eval():


    return