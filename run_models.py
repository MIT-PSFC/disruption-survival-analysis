# Functions for running all the models in Auton-Survival package
# [1] auton-survival: an Open-Source Package for Regression, Counterfactual Estimation, 
# Evaluation and Phenotyping with Censored Time-to-Event Data. arXiv (2022)

import numpy as np

# Auton-Survival models
from auton_survival.estimators import SurvivalModel # CPH, DCPH, DSM, DCM, RSF, 
from auton_survival import DeepRecurrentCoxPH # DCPH with recurrent neural network
from auton_survival.models.dsm import DeepRecurrentSurvivalMachines # DSM with recurrent neural network

# Model used in DPRF
from sklearn.ensemble import RandomForestClassifier

# Hyperparameter tuning and evaluation
from auton_survival.metrics import survival_regression_metric
from sklearn.model_selection import ParameterGrid


def run_survival_model(model_string, x_tr, x_val, y_tr, y_val):
    """
    Train and tune a SurvivalModel from auton-survival package
    
    Taken from example script in auton-survival package
    Survival Regression with Auton-Survival.ipynb
    """

    # Define the hyperparameter grid for hyperparameter tuning
    if model_string == 'cph':
        param_grid = {'l2' : [1e-3, 1e-4]}
    elif model_string == 'dcph':
        param_grid = {'bs' : [100, 200],
                'learning_rate' : [ 1e-4, 1e-3],
                'layers' : [ [100], [100, 100] ]
                }
    elif model_string == 'dsm':
        param_grid = {'layers' : [[100], [100, 100], [200]],
              'distribution' : ['Weibull', 'LogNormal'],
              'max_features' : ['sqrt', 'log2']
             }
    elif model_string == 'dcm':
        param_grid = {'k' : [2, 3],
              'learning_rate' : [1e-3, 1e-4],
              'layers' : [[100], [100, 100]]
             }
    elif model_string == 'rsf':
        param_grid = {'n_estimators' : [100, 300],
              'max_depth' : [3, 5],
              'max_features' : ['sqrt', 'log2']
             }
    else:
        raise ValueError(f"Invalid model string: {model_string}")

    params = ParameterGrid(param_grid)
    
    # Define the times for model evaluation
    # TODO What should this be? 
    times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 0.9, 10)).tolist()

    # Perform hyperparameter tuning for SurvivalModel 
    models = []
    for param in params:
        if model_string == 'cph':
            model = SurvivalModel('cph', l2=param['l2'])
            model.fit(x_tr, y_tr)
        elif model_string == 'dcph':
            model = SurvivalModel('dcph', bs=param['bs'], learning_rate=param['learning_rate'], layers=param['layers'])
            model.fit(x_tr, y_tr)
        elif model_string == 'dsm':
            model = SurvivalModel('dsm', layers=param['layers'], distribution=param['distribution'], max_features=param['max_features'])
            model.fit(x_tr, y_tr)
        elif model_string == 'dcm':
            model = SurvivalModel('dcm', k=param['k'], learning_rate=param['learning_rate'], layers=param['layers'])
            model.fit(x_tr, y_tr)
        elif model_string == 'rsf':
            model = SurvivalModel('rsf', n_estimators=param['n_estimators'], max_depth=param['max_depth'], max_features=param['max_features'])
        else:
            raise ValueError(f"Invalid model string: {model_string}")


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

def run_recurrent_model(model_string, x_tr, t_tr, e_tr, x_val, t_val, e_val):
    """
    Train and tune a recurrent model from auton-survival package
    Taken from RDSM on PBC dataset.ipynb
    """

    param_grid = {'k' : [3, 4, 6],
              'distribution' : ['LogNormal', 'Weibull'],
              'learning_rate' : [1e-4, 1e-3],
              'hidden': [50, 100],
              'layers': [3, 2, 1],
              'typ': ['LSTM', 'GRU', 'RNN']
             }

    #model = DeepRecurrentSurvivalMachines(k=param['k'], distribution=param['distribution'], hidden=param['hidden'], typ=param['typ'], layers=param['layers'])
    #model.fit(x_tr, y_tr['time'], y_tr['event'], learning_rate=param['learning_rate'])
    pass

def run_rf_model(x_tr, x_val, y_tr, y_val):
    """
    Train and tune a random forest model
    """

    param_grid = {'n_estimators' : [100, 300],
              'max_depth' : [3, 5],
              'max_features' : ['sqrt', 'log2']
             }
    
    model = RandomForestClassifier()

    pass

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
