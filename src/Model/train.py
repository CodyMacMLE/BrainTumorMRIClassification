# External Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Internal Imports


def fit(
        model: nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: str
):
    """
    Fit the model with given data.
    :param model: The neural network model to be trained.
    :param train_data: Training data in the form of a dataloader.
    :param valid_data: Validation data in the form of a dataloader.
    :param loss_fn: The loss function to be used for training.
    :param optimizer: The optimizer to be used for training.
    :param epochs: The number of epochs to train for.
    :param device: The device to be used for training (CPU or GPU).
    :return: dictionary containing training and validation loss and accuracy for each epoch. {Epoch: tuple(train_loss, valid_loss, train_accuracy, valid_accuracy)}
    """
    epoch_data = {}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device) # Move data to device

            optimizer.zero_grad() # Set gradients to zero before backpropagation
            output = model(images) # Forward pass

            loss = loss_fn(output, labels) # Calculate the loss
            loss.backward() # Backward pass (Back Propagation)
            optimizer.step() # Optimize

            train_correct += output.argmax(dim=1).eq(labels).sum().item()
            train_total += labels.size(0)
            total_train_loss += loss.item() # Accumulate loss for the epoch

        # Model evaluation on validation data
        model.eval()
        total_val_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad(): # Tells model to not handle gradients
            for images, labels in valid_data:
                images, labels = images.to(device), labels.to(device) # Move data to devic

                output = model(images)
                loss = loss_fn(output, labels)

                valid_correct += output.argmax(dim=1).eq(labels).sum().item()
                valid_total += labels.size(0)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_data) # Average loss for the epoch
        avg_train_acc = train_correct / train_total # Average accuracy for the epoch
        avg_val_loss = total_val_loss / len(valid_data)  # Average loss for the epoch
        avg_val_acc = valid_correct / valid_total  # Average accuracy for the epoch

        # Add epoch metrics
        epoch_data[epoch + 1] = (avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc)
        print(f'Epoch [{epoch + 1}/{epochs}] Loss [Train | Validation]: {avg_train_loss:.4f} | {avg_val_loss:.4f} Accuracy: {avg_train_acc:.4f} | {avg_val_acc:.4f}')

    return epoch_data