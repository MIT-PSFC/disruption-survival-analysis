from preprocess_datasets import load_dataset
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
    against the actual ip"""

    shot, times = get_transformed_shot(device, dataset, shot_number, transformer)

    ip = shot['ip'].values

    survival_times = []

    # Integrate survival probability over time to get expected time to disruption
    max_t = 2
    min_t = 0
    num_t = 100
    dt = (max_t - min_t) / num_t
    integration_times = np.linspace(min_t, max_t, num_t)

    for i in range(len(shot)):
        slice_data = shot.iloc[i]

        cumulative_survival = 0
        for time in integration_times:
            cumulative_survival += model.predict_survival(slice_data, time)[0]*dt
            if cumulative_survival > 0.5:
                survival_times.append(time)
                break
        if cumulative_survival <= 0.5:
            survival_times.append(max_t)

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


def get_transformed_shot(device, dataset, shot_number, transformer):

    data = load_dataset(device=device, dataset=dataset)

    shot = data[data['shot'] == shot_number]
    shot = shot.sort_values('time')

    times = shot['time'].values
    
    shot = shot[['ip','Wmhd','n_e','kappa','li']]

    # Transform the features
    shot = transformer.transform(shot)

    return shot, times


def plot_risk(device, dataset, shot_number, risk_time, model, transformer):
    """ Given a database and a trained model, plot the predicted risk
    over a certain time horizon against the actual ip"""
    
    shot, times = get_transformed_shot(device, dataset, shot_number, transformer)

    ip = shot['ip'].values

    # Predict the risk for each feature timeslice in shot
    risks = []
    for i in range(len(shot)):
        risks.append(model.predict_risk(shot.iloc[i], risk_time)[0])

    # get max ip
    max_ip = max(abs(ip))

    # Plot the risk against the actual ip
    fig, ax = plt.subplots()
    ax.plot(times, -ip/max_ip, label='ip')
    ax.plot(times, risks, label='risk')
    ax.set_ylim(0, 1)
    ax.set_xlabel('time')
    ax.set_ylabel('ip')
    ax.legend()

def plot_survival(device, dataset, shot_number, survival_time, model, transformer):
    """ Given a database and a trained model, plot the predicted survival probability
    over a certain time horizon against the actual ip"""
    
    shot, times = get_transformed_shot(device, dataset, shot_number, transformer)

    ip = shot['ip'].values

    # Predict the risk for each feature timeslice in shot
    risks = []
    for i in range(len(shot)):
        risks.append(model.predict_survival(shot.iloc[i], survival_time)[0])

    # get max ip
    max_ip = max(abs(ip))

    # Plot the risk against the actual ip
    fig, ax = plt.subplots()
    ax.plot(times, -ip/max_ip, label='ip')
    ax.plot(times, risks, label='risk')
    ax.set_ylim(0, 1)
    ax.set_xlabel('time')
    ax.set_ylabel('ip')
    ax.legend()




    