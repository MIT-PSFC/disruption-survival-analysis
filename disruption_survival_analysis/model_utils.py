# Functions for running and evaluating all the models in Auton-Survival package
# [1] auton-survival: an Open-Source Package for Regression, Counterfactual Estimation, 
# Evaluation and Phenotyping with Censored Time-to-Event Data. arXiv (2022)

import os
import dill

# Auton-Survival models
from auton_survival.estimators import SurvivalModel # CPH, DCPH, DSM, DCM, RSF
from auton_survival import DeepRecurrentCoxPH # DCPH with recurrent neural network
from auton_survival.models.dsm import DeepRecurrentSurvivalMachines # DSM with recurrent neural network

# Model used in DPRF
from sklearn.ensemble import RandomForestClassifier

from disruption_survival_analysis.manage_datasets import load_features_outcomes, load_features_labels


# Methods for training models

def make_model(config:dict):
    """Make a model with hyperparameters depending on the config
    
    Parameters
    ----------
    config: dict
        Dictionary of everything unique to this model.
        Should contain the model type, the metric to be evaluated, which dataset to use, and the hyperparameters
    
    Returns
    -------
    model: SurvivalModel
        The survival model to be trained
    """

    model_type = config['aa-model-type']
    if model_type == 'cph':
        l2 = config['l2']
        model = SurvivalModel(model_type, l2=l2)
    elif model_type == 'dcph':
        # Make layers a list of ints
        layer_width = config['layer_width']
        layer_depth = config['layer_depth']
        layers = [layer_width] * layer_depth
        
        batch_size = config['batch_size']
        epochs = config['epochs']
        learning_rate = config['learning_rate']
        
        model = SurvivalModel(model_type, layers=layers, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
    elif model_type == 'dcm':
        # Make layers a list of ints
        layer_width = config['layer_width']
        layer_depth = config['layer_depth']
        layers = [layer_width] * layer_depth

        batch_size = config['batch_size']
        epochs = config['epochs']
        k = config['k']
        lr = config['learning_rate']
        smoothing_factor = config['smoothing_factor']
        model = SurvivalModel(model_type, k=k, layers=layers, batch_size=batch_size, lr=lr, epochs=epochs, smoothing_factor=smoothing_factor)
    elif model_type == 'dsm':
        # Make layers a list of ints
        layer_width = config['layer_width']
        layer_depth = config['layer_depth']
        layers = [layer_width] * layer_depth

        batch_size = config['batch_size']
        distribution = config['distribution']
        epochs = config['epochs']
        learning_rate = config['learning_rate']
        temperature = config['temperature']
        
        model = SurvivalModel(model_type, 
                                layers=layers, 
                                distribution=distribution, 
                                temperature=temperature, 
                                batch_size=batch_size, 
                                learning_rate=learning_rate, 
                                epochs=epochs)
    elif model_type == 'rf':
        n_estimators = config['n_estimators'] 
        criterion = config['criterion']
        min_samples_split = config['min_samples_split']
        min_samples_leaf = config['min_samples_leaf']
        max_features = config['max_features']
        model = RandomForestClassifier(n_estimators=n_estimators,
                                        criterion=criterion,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features)
    else:
        raise ValueError(f"Model type \"{model_type}\" not recognized")

    return model

def train_survival_model(model:SurvivalModel, device, dataset_path):

    x_train, y_train = load_features_outcomes(device, dataset_path, 'train')

    model.fit(x_train, y_train)


def train_recurrent_model(model_string, x_tr, t_tr, e_tr, x_val, t_val, e_val):
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

def train_random_forest_model(model:RandomForestClassifier, device, dataset_path, disruptive_window):

    x_train, labels_train = load_features_labels(device, dataset_path, 'train', disruptive_window)
    model.fit(x_train, labels_train)

# Methods to be used by experiment utils
def get_model_for_experiment(config, experiment_type):

    model_type = config['aa-model-type']

    device = config['aa-device']
    dataset_path = config['aa-dataset-path']

    # Get the model to be used
    if experiment_type == 'val':
        # There must first be a model to validate
        model = make_model(config)

        # Fit the model to the training data
        if isinstance(model, RandomForestClassifier):
            disruptive_window = config['disruptive_window']
            train_random_forest_model(model, device, dataset_path, disruptive_window)
        elif isinstance(model, SurvivalModel):
            train_survival_model(model, device, dataset_path)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    elif experiment_type == 'test':
        # There must first be a model to test

        # Attempt to load the model by name
        model_name = name_model(config)
        try:
            model = load_model(model_name, device, dataset_path)
        except FileNotFoundError:
            # If the model is not found, make it
            model = make_model(config)

            # Fit the model to the training data
            if isinstance(model, RandomForestClassifier):
                disruptive_window = config['disruptive_window']
                train_random_forest_model(model, device, dataset_path, disruptive_window)
            elif isinstance(model, SurvivalModel):
                train_survival_model(model, device, dataset_path)
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            
            # Save the model
            save_model(model, model_name, device, dataset_path)
    else:    
        raise ValueError(f"Invalid experiment type: {experiment_type}")
    
    return model


# Methods for saving and loading models

def name_model(config):
    """Create a name for the model based on how it was trained"""

    model_type = config['aa-model-type']
    alarm_type = config['ab-alarm-type']
    metric = config['ab-metric']

    if metric == 'etint':
        time_string = f"{int(config['01_tau']*1000)}ms"
    else:
        time_string = f"{int(config['ab-required-warning-time']*1000)}ms"

    name = f"{model_type}_{alarm_type}_{metric}_{time_string}"

    return name
    
def save_model(model, model_name, device, dataset_path):
    """Save model to file"""
    model_path = 'models/' + device + '/' + dataset_path
    try:
        os.makedirs(model_path)
    except:
        pass
    model_file = model_path + '/' + model_name + '.pkl'
    dill.dump(model, open(model_file, 'wb'))
    print('Saved model to ' + model_file)

def load_model(model_name, device, dataset_path):
    """Load model from file"""
    model_file = 'models/' + device + '/' + dataset_path + '/' + model_name + '.pkl'
    with open(model_file, 'rb') as f:
        model = dill.load(f)
    print('Loaded model from ' + model_file)
    return model