import torchvision
from torchvision import transforms

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