from preprocess_datasets import load_dataset
import matplotlib.pyplot as plt

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

