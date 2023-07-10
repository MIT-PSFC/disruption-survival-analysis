# Functions for running all the models in Auton-Survival package
# [1] auton-survival: an Open-Source Package for Regression, Counterfactual Estimation, 
# Evaluation and Phenotyping with Censored Time-to-Event Data. arXiv (2022)

import dill

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

# Constants
CUTOFF_TIME = 0.5 # Cutoff time for plotting

def get_train_times(y_tr):
    """
    Get the training times for survival models
    """
    try:
        return np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 0.9, 10)).tolist()
    except:
        return np.quantile(y_tr['time'], np.linspace(0.1, 0.9, 10)).tolist()


def get_test_times(y_tr):
    """
    Get the test times for survival models
    """
    # TODO: this should be limited to under 500 ms
    return np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 0.9, 10)).tolist()


def run_survival_model(model_string, x_tr, x_val, y_tr, y_val, selection='rough'):
    """
    Train and tune a SurvivalModel from auton-survival package
    
    Taken from example script in auton-survival package
    Survival Regression with Auton-Survival.ipynb

    'rough', 'coarse', '1:1', 'fine', 'very fine'
    """

    # l2 is penalizer. float. default 1e-3
    # n_estimators is number of trees. int. default 50
    # max_depth is max depth of tree. int. default 5
    # max_features is max number of features to consider when splitting. str. default is 'sqrt', can also be 'log2'
    # layers is list of hidden layer sizes. list of ints. default [100]
    # learning rate is learning rate for optimizer. float. default 1e-3
    # bs is batch size. int. default 100
    # epochs is number of complete passes through training data. int default 50
    # distribution is distribution of survival times. str. default 'Weibull', can also be 'LogNormal'

    if selection == 'fine':
        # Define hyperparameter grids for all models to use while tuning
        l2_grid = np.logspace(-5,-2,10)
        n_estimators_grid = np.linspace(50, 500, 10).astype(int)
        max_depth_grid = np.linspace(2, 10, 5).astype(int)
        max_features_grid = ['sqrt', 'log2']
        layers_grid = [[100], [100, 100], [200]]
        learning_rate_grid = np.logspace(-5,-2,10)
        bs_grid = np.linspace(50, 500, 10).astype(int)
        epochs = 200
        distribution_grid = ['Weibull', 'LogNormal']
        temperature_grid = np.linspace(0.5,1.5,10)
        k_grid = [2, 3, 4, 5, 6]
        smoothing_factor_grid = np.logspace(-5,-2,10)
        gamma_grid = np.linspace(1, 21, 10).astype(int)
    elif selection == '1:1':
        # Define hyperparameter grids for all models to use while tuning
        l2_grid = np.logspace(-5,-2,5)
        n_estimators_grid = np.linspace(50, 500, 5).astype(int)
        max_depth_grid = np.linspace(2, 10, 5).astype(int)
        max_features_grid = ['sqrt', 'log2']
        layers_grid = [[100], [100, 100], [200]]
        learning_rate_grid = np.logspace(-5,-2,5)
        bs_grid = np.linspace(50, 500, 5).astype(int)
        epochs = 50
        distribution_grid = ['Weibull', 'LogNormal']
        temperature_grid = np.linspace(0.5,1.5,3)
        k_grid = [3]
        smoothing_factor_grid = np.logspace(-5,-2,3)
        gamma_grid = [10]
    else:
        l2_grid = [1e-3, 1e-4]
        n_estimators_grid = [100, 300]
        max_depth_grid = [3,5]
        max_features_grid = ['sqrt', 'log2']
        learning_rate_grid = [1e-3, 1e-4]
        layers_grid = [[100], [100, 100]]
        bs_grid = [50, 100]
        epochs = 50
        distribution_grid = ['Weibull', 'LogNormal']
        temperature_grid = [1.0]
        k_grid = [2, 3]
        smoothing_factor_grid = [1e-3, 1e-4]
        gamma_grid = [10]



    # Define the hyperparameter grid for hyperparameter tuning
    if model_string == 'cph':
        # l2, 
        param_grid = {'l2' : l2_grid}
    elif model_string == 'dcph':
        param_grid = {'bs' : bs_grid,
                'learning_rate' : learning_rate_grid,
                'layers' : layers_grid,
                'epochs' : [epochs]
                }
    elif model_string == 'dsm':
        param_grid = {'layers' : layers_grid,
              'distribution' : distribution_grid,
              'temperature' : temperature_grid,
              'batch_size' : bs_grid,
              'learning_rate' : learning_rate_grid,
              'epochs' : [epochs],
              'max_features' : max_features_grid,
              'k' : k_grid
             }
    elif model_string == 'dcm':
        param_grid = {'k' : k_grid,
                'layers' : layers_grid,
                'batch_size' : bs_grid,
              'learning_rate' : learning_rate_grid,
                'epochs' : [epochs],
                'smoothing_factor' : smoothing_factor_grid,
                'gamma' : gamma_grid
             }
    elif model_string == 'rsf':
        param_grid = {'n_estimators' : n_estimators_grid,
              'max_depth' : max_depth_grid,
              'max_features' : ['sqrt', 'log2']
             }
    else:
        raise ValueError(f"Invalid model string: {model_string}")

    params = ParameterGrid(param_grid)
    
    # Define the times for model evaluation
    times = get_train_times(y_tr)

    # Perform hyperparameter tuning for SurvivalModel 
    models = []
    for param in params:
        if model_string == 'cph':
            model = SurvivalModel('cph', l2=param['l2'])
        elif model_string == 'dcph':
            model = SurvivalModel('dcph', bs=param['bs'], 
                                  learning_rate=param['learning_rate'], 
                                  layers=param['layers'],
                                  epochs=param['epochs'])
        elif model_string == 'dsm':
            model = SurvivalModel('dsm', layers=param['layers'], 
                                  distribution=param['distribution'],
                                    temperature=param['temperature'],
                                    batch_size=param['batch_size'],
                                    learning_rate=param['learning_rate'],
                                    iters=param['epochs'], 
                                  max_features=param['max_features'],
                                  k=param['k'])
        elif model_string == 'dcm':
            model = SurvivalModel('dcm', k=param['k'], 
                                    batch_size=param['batch_size'],
                                    epochs=param['epochs'],
                                  learning_rate=param['learning_rate'], 
                                  layers=param['layers'],
                                  smoothing_factor=param['smoothing_factor'],
                                  gamma=param['gamma'])
        elif model_string == 'rsf':
            model = SurvivalModel('rsf', n_estimators=param['n_estimators'], 
                                  max_depth=param['max_depth'], 
                                  max_features=param['max_features'])
        else:
            raise ValueError(f"Invalid model string: {model_string}")

        try:
            model.fit(x_tr, y_tr)

            # Obtain survival probabilities for validation set and compute the Integrated Brier Score 
            predictions_val = model.predict_survival(x_val, times)
            
            # Find if predictions_val contains nan, indicating there was an issue with running the model
            if np.isnan(predictions_val).any():
                print(f"NaN in predictions_val for parameters: {param}")
            else:
                metric_val = survival_regression_metric('ibs', y_val, predictions_val, times, y_tr)
                models.append([metric_val, model])
        except:
            print(f"Error in fitting model for parameters: {param}")
            
    

    if len(models) == 0:
        raise ValueError(f"No models trained for {model_string}")

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
    
    params = ParameterGrid(param_grid)
    

    # Perform hyperparameter tuning for SurvivalModel
    models = []
    for param in params:
        try:
            model = RandomForestClassifier(n_estimators=param['n_estimators'],
                                        max_depth=param['max_depth'],
                                        max_features=param['max_features'])
            model.fit(x_tr, y_tr)

            # Get score
            score = model.score(x_val, y_val)
            models.append([score, model])
        except:
            # If there is an error, skip this model
            pass

    # Select the best model based on the best metric value computed for the validation set

    if len(models) == 0:
        print("Unable to train model")
        return None

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

    # Define the times for model testing
    times = get_test_times(y_tr)

    # Obtain survival probabilities for test set
    predictions_te = model.predict_survival(x_te, times)

    # Compute the Brier Score and time-dependent concordance index for the test set to assess model performance
    results = dict()
    results['Brier Score'] = survival_regression_metric('brs', outcomes=y_te, predictions=predictions_te, 
                                                        times=times, outcomes_train=y_tr)
    results['Concordance Index'] = survival_regression_metric('ctd', outcomes=y_te, predictions=predictions_te, 
                                                        times=times, outcomes_train=y_tr)
    
    return results, times

def save_model(model, transformer, model_name, device, dataset):
    """Save model and transformer to file"""
    model_path = 'models/' + model_name + '_' + device + '_' + dataset + '.pkl'
    dill.dump([model, transformer], open(model_path, 'wb'))
    print('Saved model to ' + model_path)

def load_model(model_name, device, dataset):
    """Load model and transformer from file"""
    model_path = 'models/' + model_name + '_' + device + '_' + dataset + '.pkl'
    with open(model_path, 'rb') as f:
        model, transformer = dill.load(f)
    print('Loaded model from ' + model_path)
    return model, transformer