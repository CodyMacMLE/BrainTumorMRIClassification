# External Imports
import torch
from torch.utils.data import DataLoader

# Internal Imports
from Typedef.Patients import Patients
from mri_dataset import MriDataset
from transforms import get_train_transforms, get_val_transforms

def get_dataloaders(
      train_data: Patients,
      val_data: Patients,
      test_data: Patients,
      batch_size: int = 32
  ) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataloader = DataLoader(MriDataset(train_data, transform=get_train_transforms()), batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(MriDataset(val_data, transform=get_val_transforms()), batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(MriDataset(test_data, transform=get_val_transforms()), batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader