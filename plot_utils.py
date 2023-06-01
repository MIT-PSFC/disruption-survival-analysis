from preprocess_datasets import load_dataset
import matplotlib.pyplot as plt

def plot_shot(device, dataset, shot_number):
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

    notused, notused, data = load_dataset(device, dataset)

    shot = data[data['shot'] == shot_number]
    shot = shot.sort_values('time')

    # Put each feature on a separate subplot
    fig, axes = plt.subplots(len(shot.columns) - 2, 1, sharex=True, figsize=(10, 10))
    for i, col in enumerate(shot.columns[2:]):
        axes[i].plot(shot['time'], shot[col])
        axes[i].set_ylabel(col)
    axes[-1].set_xlabel('time')
    fig.show()

