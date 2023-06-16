from preprocess_datasets import load_dataset, load_features_outcomes, DEFAULT_FEATURES
import matplotlib.pyplot as plt
import numpy as np

def plot_shot(device, dataset, shot_number, match_times=True):
    """ Plots the time series data for a given shot in a dataset
    Parameters
    ----------
    device : str
        The device for which to load the data
    dataset : str
        The dataset to load
    shot_number : int
        The shot number to plot
    """

    data = load_dataset(device=device, dataset=dataset)

    shot = data[data['shot'] == shot_number]
    shot = shot.sort_values('time')

    # Determine if the shot was disrupted
    disrupted = shot['time_until_disrupt'].notnull().any()

    # Put each feature on a separate subplot
    fig, axes = plt.subplots(len(shot.columns) - 4, 1, sharex=True, figsize=(10, 10))
    for i, col in enumerate(shot.columns[4:]):
        axes[i].plot(shot['time'], shot[col], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        axes[i].set_ylabel(col)
        if match_times:
            axes[i].set_xlim(data['time'].min(), data['time'].max())
    axes[-1].set_xlabel('time')

    # Add a title
    caption = '{} shot {} from {} dataset'.format(device, shot_number, dataset)
    if disrupted:
        caption += '\n(disrupted)'
    else:
        caption += ('\n(not disrupted)')
    
    fig.suptitle(caption)

    fig.show()

def plot_time_to_disrupt(device, dataset, shot_number, risk_time, model, transformer):
    """ Given features for a shot and a trained model, plot the predicted time to disruption
    against the actual ip
    
    THIS IS NOT RIGOROUS, JUST A 'WHAT IF'
    """

    shot, times = get_transformed_shot(device, dataset, shot_number, transformer)

    ip = shot['ip'].values

    survival_times = []

    # Find expected value of survival time
    max_t = 3
    min_t = 0
    num_t = 100
    test_times = np.linspace(min_t, max_t, num_t)

    for i in range(len(shot)):
        slice_data = shot.iloc[i]

        expected_survival = 0
        for time in test_times:
            expected_survival += model.predict_survival(slice_data, time)[0]

        expected_survival /= num_t
        survival_times.append(expected_survival)
        

    # get max ip
    max_ip = max(abs(ip))

    # Plot the risk against the actual ip
    fig, ax = plt.subplots()
    ax.plot(times, -ip/max_ip, label='ip')
    ax.plot(times, survival_times, label='expected time to disruption')
    ax.set_ylim(0, 1)
    ax.set_xlabel('time')
    ax.set_ylabel('ip')
    ax.legend()


def get_transformed_shot(device, dataset, shot_number, transformer, features=DEFAULT_FEATURES):

    data = load_dataset(device, dataset)

    shot = data[data['shot'] == shot_number]
    shot = shot.sort_values('time')

    # Determine if the shot was disrupted
    disruptive = shot['time_until_disrupt'].notnull().any()

    times = shot['time'].values
    
    shot = shot[features]

    # Transform the features
    shot = transformer.transform(shot)

    return shot, times, disruptive


def plot_survival(device, dataset, shot_number, survival_time, models, names, transformer):
    """ Given a database and a trained model, plot the predicted survival probability
    over a certain time horizon against the actual ip"""
    
    shot, times, disruptive = get_transformed_shot(device, dataset, shot_number, transformer)

    ip = shot['ip'].values

    # For each model, plot the predicted survival probability at survival time
    # against the actual ip
    # make distinct lines on same plot
    fig, ax = plt.subplots()

    # Get normalization factor to ensure ip is always between 0 and 1
    if (-min(ip) > max(ip)):
        ip = -ip

    ax.plot(times, ip/max(ip), label='ip')
    for i, model in enumerate(models):
        # If model is an instance of SurvivalModel, use predict_survival
        # Otherwise, use predict_proba
        try:
            survival = []
            if hasattr(model, 'predict_survival'):
                for j in range(len(shot)):
                    try:
                        survival.append(model.predict_survival(shot.iloc[j], survival_time)[0])
                    except:
                        survival.append(model.predict_survival(shot.iloc[j], [survival_time])[0])
            else:
                survival = model.predict_proba(shot)[:, 0]
        except:
            # Hack to stop RSF from crashing
            survival = model.predict_survival(shot, survival_time)

        ax.plot(times, survival, label=names[i])

    ax.set_ylim(0, 1)
    ax.set_xlabel('time')
    ax.set_ylabel(f'ip and {survival_time} [s] survival probability')
    ax.legend()
    if disruptive:
        fig.suptitle(f"Survival of Shot {shot_number} from {dataset} dataset on {device} (disrupted)")
    else:
        fig.suptitle(f"Survival of Shot {shot_number} from {dataset} dataset on {device} (not disrupted)")
    fig.show()
    


def plot_risk(device, dataset, shot_number, survival_time, models, names, transformer):
    """ Given a database and a trained model, plot the predicted survival probability
    over a certain time horizon against the actual ip"""
    
    shot, times, disruptive = get_transformed_shot(device, dataset, shot_number, transformer)

    ip = shot['ip'].values

    # For each model, plot the predicted survival probability at survival time
    # against the actual ip
    # make distinct lines on same plot
    fig, ax = plt.subplots()

    # Get normalization factor to ensure ip is always between 0 and 1
    if (-min(ip) > max(ip)):
        ip = -ip

    ax.plot(times, ip/max(ip), label='ip')
    for i, model in enumerate(models):
        # If model is an instance of SurvivalModel, use predict_survival
        # Otherwise, use predict_proba
        try:
            survival = []
            if hasattr(model, 'predict_risk'):
                for j in range(len(shot)):
                    try:
                        survival.append(model.predict_risk(shot.iloc[j], survival_time)[0])
                    except:
                        survival.append(model.predict_risk(shot.iloc[j], np.array([survival_time]))[0])
            else:
                survival = model.predict_proba(shot)[:, 1]
        except:
            # Hack to stop RSF from crashing
            survival = model.predict_risk(shot, survival_time)

        ax.plot(times, survival, label=names[i])

    ax.set_ylim(0, 1)
    ax.set_xlabel('time')
    ax.set_ylabel(f'ip and {survival_time} [s] risk')
    ax.legend()
    if disruptive:
        fig.suptitle(f"Risk of Shot {shot_number} from {dataset} dataset on {device} (disrupted)")
    else:
        fig.suptitle(f"Risk of Shot {shot_number} from {dataset} dataset on {device} (not disrupted)")
    fig.show()
    