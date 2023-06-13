# Functions for running all the models in Auton-Survival package
# [1] auton-survival: an Open-Source Package for Regression, Counterfactual Estimation, 
# Evaluation and Phenotyping with Censored Time-to-Event Data. arXiv (2022)

import numpy as np

from auton_survival.estimators import SurvivalModel # CPH, DCPH, DSM, DCM, RSF, 
from auton_survival import DeepRecurrentCoxPH # DCPH with recurrent neural network
from auton_survival.models.dsm import DeepRecurrentSurvivalMachines # DSM with recurrent neural network

# Hyperparameter tuning and evaluation
from auton_survival.metrics import survival_regression_metric
from sklearn.model_selection import ParameterGrid


def hyperparam_tuning(model_string, params, x_tr, x_val, y_tr, y_val):
    """
    Perform hyperparameter tuning for a given model
    
    Taken from example script in auton-survival package
    Survival Regression with Auton-Survival.ipynb
    """
    
    # Define the times for model evaluation
    # TODO What should this be? 
    times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 0.9, 10)).tolist()

    if model_string in ['cph', 'dcph', 'dsm', 'dcm', 'rsf', 'drsm']:
        
        # Perform hyperparameter tuning for SurvivalModel 
        models = []
        for param in params:
            if model_string == 'cph':
                model = SurvivalModel('cph', l2=param['l2'])
            elif model_string == 'dcph':
                model = SurvivalModel('dcph', bs=param['bs'], learning_rate=param['learning_rate'], layers=param['layers'])
            elif model_string == 'dsm':
                model = SurvivalModel('dsm', layers=param['layers'], distribution=param['distribution'], max_features=param['max_features'])
            elif model_string == 'dcm':
                model = SurvivalModel('dcm', k=param['k'], learning_rate=param['learning_rate'], layers=param['layers'])
            elif model_string == 'rsf':
                model = SurvivalModel('rsf', n_estimators=param['n_estimators'], max_depth=param['max_depth'], max_features=param['max_features'])
            elif model_string == 'drsm':
                model = DeepRecurrentSurvivalMachines(k=param['k'], distribution=param['distribution'], hidden=param['hidden'], typ=param['typ'], layers=param['layers'])
            else:
                raise ValueError(f"Invalid model string: {model_string}")

            # The fit method is called to train the model
            if model_string == 'drsm':
                model.fit(x_tr, y_tr['time'], y_tr['event'], learning_rate=param['learning_rate'])
            else:
                model.fit(x_tr, y_tr)

            # Obtain survival probabilities for validation set and compute the Integrated Brier Score 
            predictions_val = model.predict_survival(x_val, times)
            
            # Find if predictions_val contains nan, indicating there was an issue with running the model
            if np.isnan(predictions_val).any():
                print(f"NaN in predictions_val for parameters: {param}")
            else:
                metric_val = survival_regression_metric('ibs', y_val, predictions_val, times, y_tr)
                models.append([metric_val, model])
            
        # Select the best model based on the mean metric value computed for the validation set
        metric_vals = [i[0] for i in models]
        first_min_idx = metric_vals.index(min(metric_vals))
        model = models[first_min_idx][1]

        return model


def eval_model(model, x_te, y_tr, y_te):
    """
    Evaluate Cox Survival model
    Taken from example script in auton-survival package
    Survival Regression with Auton-Survival.ipynb
    """    

    # Define the times for model evaluation
    # TODO What should this be?
    times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 0.9, 10)).tolist()

    # Obtain survival probabilities for test set
    predictions_te = model.predict_survival(x_te, times)

    # Compute the Brier Score and time-dependent concordance index for the test set to assess model performance
    results = dict()
    results['Brier Score'] = survival_regression_metric('brs', outcomes=y_te, predictions=predictions_te, 
                                                        times=times, outcomes_train=y_tr)
    results['Concordance Index'] = survival_regression_metric('ctd', outcomes=y_te, predictions=predictions_te, 
                                                        times=times, outcomes_train=y_tr)
    
    return results, times


def run_cph(x_tr, x_val, y_tr, y_val):
    """
    Run Cox Proportional Hazards model

    Parameters
    ----------
    x_tr : numpy array
        Training set features
    x_val : numpy array
        Validation set features
    y_tr : numpy array
        Training set outcomes
    y_val : numpy array
        Validation set outcomes
    """

    # Define parameters for tuning the model
    param_grid = {'l2' : [1e-3, 1e-4]}
    params = ParameterGrid(param_grid)

    model = hyperparam_tuning('cph', params, x_tr, x_val, y_tr, y_val)

    return model

def run_dcph(x_tr, x_val, y_tr, y_val):
    """
    Run Deep Cox Proportional Hazards model

    Parameters
    ----------
    x_tr : numpy array
        Training set features
    x_val : numpy array
        Validation set features
    y_tr : numpy array
        Training set outcomes
    y_val : numpy array
        Validation set outcomes
    """

    # Define parameters for tuning the model
    # Define parameters for tuning the model
    param_grid = {'bs' : [100, 200],
                'learning_rate' : [ 1e-4, 1e-3],
                'layers' : [ [100], [100, 100] ]
                }
    params = ParameterGrid(param_grid)

    model = hyperparam_tuning('dcph', params, x_tr, x_val, y_tr, y_val)

    return model

def run_dsm(x_tr, x_val, y_tr, y_val):

    # Define parameters for tuning the model
    param_grid = {'layers' : [[100], [100, 100], [200]],
              'distribution' : ['Weibull', 'LogNormal'],
              'max_features' : ['sqrt', 'log2']
             }
    params = ParameterGrid(param_grid)

    model = hyperparam_tuning('dsm', params, x_tr, x_val, y_tr, y_val)

    return model

def run_dcm(x_tr, x_val, y_tr, y_val):

    # Define parameters for tuning the model
    param_grid = {'k' : [2, 3],
              'learning_rate' : [1e-3, 1e-4],
              'layers' : [[100], [100, 100]]
             }
    params = ParameterGrid(param_grid)

    model = hyperparam_tuning('dcm', params, x_tr, x_val, y_tr, y_val)

    return model

def run_rsf(x_tr, x_val, y_tr, y_val):

    # Define parameters for tuning the model
    param_grid = {'n_estimators' : [100, 300],
              'max_depth' : [3, 5],
              'max_features' : ['sqrt', 'log2']
             }

    params = ParameterGrid(param_grid)

    model = hyperparam_tuning('rsf', params, x_tr, x_val, y_tr, y_val)

    return model

def run_drsm(x_tr, x_val, y_tr, y_val):

    param_grid = {'k' : [3, 4, 6],
              'distribution' : ['LogNormal', 'Weibull'],
              'learning_rate' : [1e-4, 1e-3],
              'hidden': [50, 100],
              'layers': [3, 2, 1],
              'typ': ['LSTM', 'GRU', 'RNN']
             }
    params = ParameterGrid(param_grid)

    model = hyperparam_tuning('drsm', params, x_tr, x_val, y_tr, y_val)

    return model

