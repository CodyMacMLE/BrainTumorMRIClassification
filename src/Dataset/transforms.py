import torch
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms


"""
First two transform classes are used for segmentation U-net
"""
class Transforms(torch.nn.Module):
    def __init__(self, pixel_transforms = False):
        super().__init__()
        self.transform_pixels = pixel_transforms

        self.spatial_transformers = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=360),
        ])

        if self.transform_pixels:
            self.pixel_transformers = PixelTransforms()

        self.final_transformers = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, image, mask = None):
        if mask is not None:
            mask = tv_tensors.Mask(mask)
            image, mask = self.spatial_transformers(image, mask)
        else:
            image = self.spatial_transformers(image)

        if self.transform_pixels:
            image = self.pixel_transformers(image)

        image = self.final_transformers(image)

        return image, mask


class PixelTransforms(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        ])

    def forward(self, image):
        image = self.transforms(image)
        return image


"""
Below are used in the resnet and baseline cnn models
"""
def get_train_transforms() -> torchvision.transforms.Compose:
    """
    Augmentation transforms: Random rotations, flips and brightness/contrast jitter
    :return:
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=360),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform


def get_val_transforms() -> torchvision.transforms.Compose:
    """
    Non-augmentation transforms: Resizes for ResNet (224x224), converts to a tensor, normalizes pixel values
    :return:
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform