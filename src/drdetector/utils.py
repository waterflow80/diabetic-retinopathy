import os

import matplotlib.pyplot as plt

def plot_loss_curves(train_losses, val_losses, filename, plot_dir):
    """
    Plots training and validation loss curves and saves the plot.

    Parameters:
    ----------
    :param train_losses:
        training loss values over epochs.
    :type train_losses: list | np.array
    :param val_losses:
        Validation loss values over epochs.
    :type val_losses: list | np.array
    :param filename:
        Name of the model's considering backbone used, included in the plot title.
    :type filename: str
    :param plot_dir:
        Directory path to save the plotted loss curves.
    :type plot_dir: str
    :return:
    """
    epochs = range(1, len(train_losses) + 1)
    # Plot losses
    plt.figure(figsize=(10, 2))
    plt.plot(epochs, train_losses, label='Train loss')
    plt.plot(epochs, val_losses, label='Val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{filename}.png"))


def get_models_list(models_path="/models"):
    ...