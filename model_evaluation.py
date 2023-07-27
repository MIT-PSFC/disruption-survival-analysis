# Different methods of evaluating the model to have good hyperparameter tuning
import numpy as np
from auton_survival.estimators import SurvivalModel # CPH, DCPH, DSM, DCM, RSF
from auton_survival.preprocessing import Preprocessor
from auton_survival.metrics import survival_regression_metric
from manage_datasets import load_features_outcomes, load_feature_list

def get_val_times(y_tr, min_quantile, max_quantile, num_quantiles):
    """
    Get the validation times for survival models
    """
    return np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(min_quantile, max_quantile, num_quantiles)).tolist()

def load_timeslice_data(device, dataset_path, valmin, valmax, numval):
    """ Load the training and validation data on a per-timeslice basis

    Parameters:
    ----------
    device : str
        The device to train on.
    dataset_path : str
        The path to the dataset.
    valmin : float
        The minimum time quantile to validate on.
    valmax : float
        The maximum time quantile to validate on.
    numval : int
        The number of time quantiles to validate on.
    """

    # Load the training and validation data

    numeric_feats = load_feature_list(device, dataset_path)

    features_train, outcomes_train = load_features_outcomes(device, dataset_path, 'train', numeric_feats)
    features_val, outcomes_val= load_features_outcomes(device, dataset_path, 'val', numeric_feats)

    # Fit the imputer and scaler to the training data and transform the training, test, and validation data
    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
    transformer=preprocessor.fit(features_train, cat_feats=[], num_feats=numeric_feats, one_hot=True, fill_value=-1)

    x_train = transformer.transform(features_train)
    y_train = outcomes_train
    x_val = transformer.transform(features_val)
    y_val = outcomes_val

    val_times = get_val_times(y_train, valmin, valmax, numval)

    return x_train, y_train, x_val, y_val, val_times

def evaluate_model(device, dataset_path, model, evaluation_method, valmin, valmax, numval):

    if evaluation_method == 'timeslice_ibs':
        metric_val = timeslice_eval(device, dataset_path, model, valmin, valmax, numval)
    elif evaluation_method == 'missed_alarm':
        #metric_val = missed_alarm_eval(model, device, dataset_path)
        metric_val = None
    else:
        metric_val = None

    return metric_val

def timeslice_eval(device, dataset_path, model, valmin, valmax, numval):

    # Load the training and validation data
    x_train, y_train, x_val, y_val, val_times = load_timeslice_data(device, dataset_path, valmin, valmax, numval)

    # Fit the model
    try:
        model.fit(x_train, y_train)

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