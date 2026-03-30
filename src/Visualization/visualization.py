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
        plt.show()

def visualize_segment_mask(original, predicted, ground_truth, overlaid = False, alpha = 0.5):
    original = original.resize((224, 224))
    ground_truth = ground_truth.resize((224, 224))

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(15, 5)

    fig.suptitle("MRI Segment Visualization")
    axs[0].xaxis.set_visible(False)
    axs[0].yaxis.set_visible(False)
    axs[1].xaxis.set_visible(False)
    axs[1].yaxis.set_visible(False)
    axs[2].xaxis.set_visible(False)
    axs[2].yaxis.set_visible(False)

    axs[0].set_title("Original")
    axs[0].imshow(original)

    axs[1].set_title("Predicted Mask")
    if overlaid:
        axs[1].imshow(original)
        axs[1].imshow(predicted, alpha=alpha, cmap='jet')
    else:
        axs[1].imshow(predicted, cmap='gray')

    axs[2].set_title("Ground Truth Mask")
    axs[2].imshow(ground_truth, cmap='gray')
