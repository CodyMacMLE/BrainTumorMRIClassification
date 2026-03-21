import torch
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
import numpy as np

def evaluate(model: torch.nn.Module, test_data: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    results = ([],[])

    model.eval()
    with torch.no_grad():
        for images, labels in test_data:
            images, labels = images.to(device), labels.to(device)  # Move data to device

            output = model(images) # forward pass
            pred = torch.argmax(output, dim=1) # get pred

            results[0].append(pred.cpu())
            results[1].append(labels.cpu())

    return torch.cat(results[0]).numpy(), torch.cat(results[1]).numpy()