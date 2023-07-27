# Different methods of evaluating the model to have good hyperparameter tuning

def load_data(device, dataset_path, valmin, valmax, numval):

    # Load the training and validation data

    return x_train, y_train, x_val, y_val, val_times

def evaluate_model(device, dataset_path, model, evaluation_method, valmin, valmax, numval):

    # Load the training and validation data

    if evaluation_method == 'timeslice':
        metric_val = timeslice_eval(model, device, dataset_path)
    elif evaluation_method == 'missed_alarm':
        metric_val = missed_alarm_eval(model, device, dataset_path)

    return metric_val

def timeslice_eval(model, device, dataset_path):


    return metric_val

def missed_alarm_eval():


    return

def disruptive_shot_expected_lifetime_eval():


    return