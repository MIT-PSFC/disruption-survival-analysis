# Functions for running all the models in Auton-Survival package
# [1] auton-survival: an Open-Source Package for Regression, Counterfactual Estimation, 
# Evaluation and Phenotyping with Censored Time-to-Event Data. arXiv (2022)

import numpy as np

from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
from sklearn.model_selection import ParameterGrid

def run_cph(x_tr, x_val, y_tr, y_val):
    """
    Run Cox Proportional Hazards model
    Taken from example script in auton-survival package
    Survival Regression with Auton-Survival.ipynb

    Parameters
    ----------
    x_tr : numpy array
        Training set features
    x_te : numpy array
        Test set features
    x_val : numpy array
        Validation set features
    """

    # Define parameters for tuning the model
    param_grid = {'l2' : [1e-3, 1e-4]}
    params = ParameterGrid(param_grid)

    # Define the times for model evaluation
    # TODO should this be tuned?
    times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 0.9, 10)).tolist()

    # Perform hyperparameter tuning 
    models = []
    for param in params:
        model = SurvivalModel('cph', random_seed=2, l2=param['l2'])
        
        # The fit method is called to train the model
        model.fit(x_tr, y_tr)

        # Obtain survival probabilities for validation set and compute the Integrated Brier Score 
        predictions_val = model.predict_survival(x_val, times)
        metric_val = survival_regression_metric('ibs', y_val, predictions_val, times, y_tr)
        models.append([metric_val, model])
        
    # Select the best model based on the mean metric value computed for the validation set
    metric_vals = [i[0] for i in models]
    first_min_idx = metric_vals.index(min(metric_vals))
    model = models[first_min_idx][1]

    return model


def eval_cph(model, x_te, y_tr, y_te):
    """
    Evaluate Cox Proportional Hazards model
    Taken from example script in auton-survival package
    Survival Regression with Auton-Survival.ipynb

    """    

    # Define the times for model evaluation
    # TODO should this be tuned?
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

def run_cph_

