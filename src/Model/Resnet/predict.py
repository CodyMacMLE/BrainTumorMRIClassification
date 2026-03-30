from pathlib import Path
import numpy as np
import PIL
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
import torch.nn as nn
import os
from numpy.typing import NDArray

def predict(model: nn.Module, image: os.PathLike | NDArray, device: str):
    from .GradCAM import GradCAM
    from src.Dataset.transforms import get_val_transforms

    # Prepare the model
    model.to(device)

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.eval()

    grad_cam_hooks = GradCAM(model, target_layer="layer4.1.conv2")

    # Prepare the Image
    if isinstance(image, np.ndarray):
        image_transformed = Image.fromarray(image)
    elif isinstance(image, os.PathLike):
        image_transformed = Image.open(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    transformers = get_val_transforms()
    image_transformed = transformers(image_transformed)
    image_transformed = image_transformed.unsqueeze(0).to(device)

    # Generate output
    cam, prediction = grad_cam_hooks.generate(image_transformed)

    pred_class = prediction.argmax(dim=1).item()
    confidence = prediction.max(dim=1).values.item() * 100

    return pred_class, confidence, cam