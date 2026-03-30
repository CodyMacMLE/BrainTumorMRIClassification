# External Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from datetime import datetime
import torch.optim.lr_scheduler as lr_scheduler

# Internal Imports
from src.Model.Unet.Unet import DiceScore

# Color constants
BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def fit(
        model: nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        epochs: int = 1000,
        patience: int = 5,
        segmentation: bool = False
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
    :param patience: The number of epochs to wait for improvement before early stopping.
    :return: dictionary containing training and validation loss and accuracy for each epoch. {Epoch: tuple(train_loss, valid_loss, train_accuracy, valid_accuracy)}
    """
    epoch_data = {}
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    best_epoch = 0
    dice_score = DiceScore()

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//2)

    for epoch in range(epochs):
        start = datetime.now()
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_dice_score = 0.0

        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device) # Move data to device

            optimizer.zero_grad() # Set gradients to zero before backpropagation
            output = model(images) # Forward pass

            if segmentation:
                output = torch.sigmoid(output)
                thresh_output = (output > 0.5).float()  # Thresholding to get binary mask
                score = dice_score(thresh_output, labels)
                train_dice_score += score.item()

            loss = loss_fn(output, labels) # Calculate the loss
            loss.backward() # Backward pass (Back Propagation)
            optimizer.step() # Optimize

            if not segmentation:
                train_correct += output.argmax(dim=1).eq(labels).sum().item()
                train_total += labels.size(0)

            total_train_loss += loss.item() # Accumulate loss for the epoch

        # Model evaluation on validation data
        model.eval()
        total_val_loss = 0.0
        valid_correct = 0
        valid_total = 0
        val_dice_score = 0.0

        with torch.no_grad(): # Tells model to not handle gradients
            for images, labels in valid_data:
                images, labels = images.to(device), labels.to(device) # Move data to devic

                output = model(images)

                if segmentation:
                    output = torch.sigmoid(output)
                    thresh_output = (output > 0.5).float() # Thresholding to get binary mask
                    score = dice_score(thresh_output, labels)
                    val_dice_score += score.item()

                    loss = loss_fn(output, labels)
                    total_val_loss += loss.item()
                else:
                    loss = loss_fn(output, labels)
                    valid_correct += output.argmax(dim=1).eq(labels).sum().item()
                    valid_total += labels.size(0)

                    total_val_loss += loss.item()

        if segmentation:
            avg_train_acc = train_dice_score / len(train_data)
            avg_val_acc = val_dice_score / len(valid_data)
        else:
            avg_train_acc = train_correct / train_total
            avg_val_acc = valid_correct / valid_total


        avg_train_loss = total_train_loss / len(train_data) # Average loss for the epoch
        avg_val_loss = total_val_loss / len(valid_data)  # Average loss for the epoch

        scheduler.step(avg_val_loss)

        # Add epoch metrics
        epoch_data[epoch + 1] = (avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc)
        formatted_epoch = f'{epoch + 1}'.rjust(3, ' ')
        alpha = optimizer.param_groups[0]['lr']

        print(
            f"\n{BOLD}{GREEN}Epoch {formatted_epoch}/{epochs}{RESET} | "
            f"Time: {str(datetime.now() - start).split('.')[0]} | "
            f"Patience: {patience_counter}/{patience} | "
            f"LR: {alpha}\n"
            f"  {BLUE}Train{RESET}  — Loss: {avg_train_loss:.4f}  Dice: {avg_train_acc:.4f}\n"
            f"  {YELLOW}Valid{RESET}  — Loss: {avg_val_loss:.4f}  Dice: {avg_val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter == patience:
            print(f"[INFO] Patience reached, best epoch was {best_epoch} with {best_val_loss:.4f} validation loss.")
            break

    model.load_state_dict(best_weights)
    return epoch_data