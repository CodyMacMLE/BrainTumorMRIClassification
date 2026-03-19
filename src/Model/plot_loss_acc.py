# External Imports
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import os

def plot_loss_acc(epoch_data, title="Loss and Accuracy Curves", path: os.PathLike = None):
    train_loss, val_loss, train_acc, val_acc = zip(*epoch_data.values())

    # Display in matplotlib
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)

    axs[0].plot(train_loss, label="Train Loss")
    axs[0].plot(val_loss, label="Validation Loss")
    axs[0].set_title("Loss Curves")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(train_acc, label="Train Accuracy")
    axs[1].plot(val_acc, label="Validation Accuracy")
    axs[1].set_title("Accuracy Curves")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    if path:
        fig.savefig(path)
    else:
        path = Path(__file__).resolve().parent.parent.parent / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{path}/loss-curves_{datetime.now().strftime('%Y-%m-%d_%Hh%m-%Ss')}.png")

    plt.show()