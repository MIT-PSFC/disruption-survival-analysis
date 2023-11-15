import os
import sys
import yaml
import optuna

from disruption_survival_analysis.Experiments import Experiment

def objective(trial, sweep_config):

    experiment_config = {}

    # Copy some relevant values into the experiment config
    experiment_config["device"] = sweep_config["device"]
    experiment_config["dataset_path"] = sweep_config["dataset_path"]

    experiment_config["model_type"] = sweep_config["model_type"]
    experiment_config["alarm_type"] = sweep_config["alarm_type"]
    experiment_config["metric"] = sweep_config["metric"]
    experiment_config["required_warning_time"] = sweep_config["required_warning_time"]
    experiment_config["hyperparameters"] = {}

    hyperparameters = sweep_config["hyperparameters"]
    # For each hyperparameter in the sweep config, pick a value for the trial
    hyperparameter_names = list(hyperparameters.keys())

    for hyperparameter_name in hyperparameter_names:
        hyperparameter = hyperparameters[hyperparameter_name]
        
        # Get the distribution type
        distribution_type = hyperparameter["distribution"]

        # Get the value for the trial
        if distribution_type == "categorical":
            choices = hyperparameter["values"]
            value = trial.suggest_categorical(hyperparameter_name, choices)
        elif distribution_type == "int":
            min = hyperparameter["min"]
            max = hyperparameter["max"]
            value = trial.suggest_int(hyperparameter_name, min, max)
        elif distribution_type == "float":
            min = hyperparameter["min"]
            max = hyperparameter["max"]
            log = hyperparameter["log"]
            value = trial.suggest_float(hyperparameter_name, min, max, log=log)
        else:
            raise ValueError(f"Invalid distribution type: {distribution_type}")
        
        experiment_config["hyperparameters"][hyperparameter_name] = value

    try:
        # Create the experiment and try to get the evaluation metric
        experiment = Experiment(experiment_config, 'val')
        metric_val = experiment.evaluate_metric(sweep_config["metric"])
    except Exception as e:
        print("Error during training or validation!")
        print(e)
        # If anything goes wrong during training or validation, say that the trial was pruned
        # This will cause optuna to avoid hyperparameters that cause errors
        raise optuna.TrialPruned()

    return metric_val

if __name__ == "__main__":
    # Get the sweep config file path from the command line
    sweep_config_path = sys.argv[1]

    # Load the sweep config
    sweep_config = yaml.safe_load(open(f"models/{sweep_config_path}", "r"))

    # Get the path to put the study database
    # Since SQLite can't accept writes from more than one process at a time,
    # each sweep gets its own database
    device = sweep_config["device"]
    dataset_path = sweep_config["dataset_path"]
    model_type = sweep_config["model_type"]
    alarm_type = sweep_config["alarm_type"]
    metric = sweep_config["metric"]
    required_warning_time = int(sweep_config["required_warning_time"]*1000)
    database_path = f"models/{device}/{dataset_path}/studies/{model_type}_{alarm_type}_{metric}_{required_warning_time}ms_study.db"

    # Get the direction the study should be optimized in
    if metric in ["auroc", "auwtc", "maxf1"]:
        direction = "maximize"
    else:
        direction = "minimize"

    # Create database folder if it doesn't exist yet
    database_folder = os.path.dirname(database_path)
    if not os.path.exists(database_folder):
        try:
            os.makedirs(database_folder)
        except:
            pass

    lock_obj = optuna.storages.JournalFileOpenLock(database_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(database_path, lock_obj=lock_obj)
    )
    
    study = optuna.create_study(
        storage=storage, 
        study_name=f"{model_type}_{alarm_type}_{metric}_{required_warning_time}ms",
        direction=direction,
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, sweep_config), n_trials=10)