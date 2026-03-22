from pathlib import Path
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
import torch.nn as nn
import os

def predict(model: nn.Module, image: os.PathLike, device: str):
    from .GradCAM import GradCAM
    from src.Dataset.transforms import get_val_transforms

    # Prepare the model
    model.to(device)

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.eval()

    grad_cam_hooks = GradCAM(model, target_layer="layer4.1.conv2")

    # Prepare the Image
    image_transformed = Image.open(image)
    transformers = get_val_transforms()
    image_transformed = transformers(image_transformed)
    image_transformed = image_transformed.unsqueeze(0).to(device)

    # Generate output
    cam, prediction = grad_cam_hooks.generate(image_transformed)

    return prediction, cam