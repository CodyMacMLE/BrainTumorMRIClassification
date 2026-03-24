# External Imports
from torchvision import models
from torch import nn as nn

def build_resnet(num_classes):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in resnet.parameters():
        param.requires_grad = False

    resnet.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512, out_features=num_classes)
    )
    return resnet