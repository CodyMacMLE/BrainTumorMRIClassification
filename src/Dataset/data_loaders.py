# External Imports
# Internal Imports
from Typedef.Patients import Patients
from torch.utils.data import DataLoader

from .mri_dataset import MriDataset


def get_dataloaders(
        train_data: Patients,
        val_data: Patients,
        test_data: Patients,
        batch_size: int = 32,
        segmentation = False
  ) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_dataloader = DataLoader(MriDataset(train_data, set_type = "train", segmentation=segmentation), batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(MriDataset(val_data, set_type = "validation", segmentation=segmentation), batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(MriDataset(test_data, set_type = "test", segmentation=segmentation), batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader