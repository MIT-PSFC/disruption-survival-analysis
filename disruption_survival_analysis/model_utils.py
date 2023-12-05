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

from disruption_survival_analysis.manage_datasets import load_features_outcomes, load_features_labels, print_memory_usage


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

    model_type = config['model_type']

    hyperparameters = config['hyperparameters']
    if model_type == 'cph':
        l2 = hyperparameters['l2']
        model = SurvivalModel(
            model_type, 
            l2=l2
        )
    elif model_type == 'dcph':
        layer_width = hyperparameters['layer_width']
        layer_depth = hyperparameters['layer_depth']
        layers = [layer_width] * layer_depth
        batch_size = hyperparameters['batch_size']
        epochs = hyperparameters['epochs']
        learning_rate = hyperparameters['learning_rate']
        
        model = SurvivalModel(
            model_type, 
            layers=layers, 
            learning_rate=learning_rate, 
            batch_size=batch_size, 
            epochs=epochs
        )
    elif model_type == 'dcm':
        layer_width = hyperparameters['layer_width']
        layer_depth = hyperparameters['layer_depth']
        layers = [layer_width] * layer_depth
        batch_size = hyperparameters['batch_size']
        epochs = hyperparameters['epochs']
        k = hyperparameters['k']
        lr = hyperparameters['learning_rate']
        smoothing_factor = hyperparameters['smoothing_factor']

        model = SurvivalModel(
            model_type, 
            k=k, 
            layers=layers, 
            batch_size=batch_size, 
            lr=lr, 
            epochs=epochs, 
            smoothing_factor=smoothing_factor
        )
    elif model_type == 'dsm':
        layer_width = hyperparameters['layer_width']
        layer_depth = hyperparameters['layer_depth']
        layers = [layer_width] * layer_depth
        batch_size = hyperparameters['batch_size']
        distribution = hyperparameters['distribution']
        epochs = hyperparameters['epochs']
        learning_rate = hyperparameters['learning_rate']
        temperature = hyperparameters['temperature']
        
        model = SurvivalModel(
            model_type, 
            layers=layers, 
            distribution=distribution, 
            temperature=temperature, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            epochs=epochs
        )
    elif model_type == 'rf' or model_type == 'km':
        criterion = hyperparameters['criterion']
        max_features = hyperparameters['max_features']
        n_estimators = hyperparameters['n_estimators'] 
        min_samples_leaf = hyperparameters['min_samples_leaf']
        min_samples_split = hyperparameters['min_samples_split']

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=0
        )
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

def train_random_forest_model(model:RandomForestClassifier, device, dataset_path, class_time):

    x_train, labels_train = load_features_labels(device, dataset_path, 'train', class_time)
    model.fit(x_train, labels_train)

# Methods to be used by experiment utils
def get_model_for_experiment(config, experiment_type, training_data=None):
    """
    Get the model to be used for the experiment.

    Parameters
    ----------
    config: dict
        Dictionary of everything unique to this model.
        Should contain the model type, the metric to be evaluated, which dataset to use, and the hyperparameters
    experiment_type: str
        The type of experiment to get the model for.
        If 'val', then the model will be trained and returned.
        If 'test', the model will first attempt to be loaded, but if it does not exist it will be trained, saved, and returned.

    Returns
    -------
    model:
        The model to be used for the experiment
    """

    device = config['device']
    dataset_path = config['dataset_path']
    model_type = config['model_type']
    hyperparameters = config['hyperparameters']

    model_name = name_model(config)

    if experiment_type == 'test':
        try:
            model = load_model(model_name, device, dataset_path)
            return model
        except FileNotFoundError:
            # If the model does not exist, it must be trained
            pass

    model = make_model(config)

    # Fit the model to the training data
    if isinstance(model, RandomForestClassifier):
        class_time = hyperparameters['class_time']
        train_random_forest_model(model, device, dataset_path, class_time)
    elif isinstance(model, SurvivalModel):
        train_survival_model(model, device, dataset_path)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Save the model for testing future use
    if experiment_type == 'test':
        save_model(model, model_name, device, dataset_path)

    return model

# Methods for saving and loading models

def name_model(config):
    """Create a name for the model based on how it was trained"""

    model_type = config['model_type']
    alarm_type = config['alarm_type']
    metric = config['metric']

    if metric == 'etint':
        time_string = f"{int(config['01_tau']*1000)}ms"
    else:
        time_string = f"{int(config['required_warning_time']*1000)}ms"

    name = f"{model_type}_{alarm_type}_{metric}_{time_string}"

    return name
    
def save_model(model, model_name, device, dataset_path):
    """Save model to file"""
    model_path = f"results/{device}/{dataset_path}/models"
    try:
        os.makedirs(model_path)
    except:
        pass
    model_file = f"{model_path}/{model_name}.pkl"
    dill.dump(model, open(model_file, 'wb'))
    print('Saved model to ' + model_file)

def load_model(model_name, device, dataset_path):
    """Load model from file"""
    model_file = f"results/{device}/{dataset_path}/models/{model_name}.pkl"
    with open(model_file, 'rb') as f:
        model = dill.load(f)
    print('Loaded model from ' + model_file)
    return model

