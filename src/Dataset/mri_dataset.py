# External Imports
import torch
from PIL import Image
import numpy as np

# Internal Imports
from Typedef.Patients import Patients

class MriDataset(torch.utils.data.Dataset):
    def __init__(self, data: Patients, transform=None):
        """
        Initializes the MRI Dataset to use in training, testing, and validation
        :param data: The data of patients that have passed data integrity. The data needs to have been split
        prior to this step and given as the Patients datatype.
        :param transform:
        """

        self.mri_dataset = []
        self.transform = transform

        # Iterate through patients dict
        for _ , mri_segments in data.items():
            # Iterate through the segments list
            for mri_segment in mri_segments:
                mri_image_path, mri_mask_path = mri_segment
                # create label
                mri_mask_img = Image.open(mri_mask_path).convert('L')
                if np.array(mri_mask_img).max() > 0:
                    label = 1
                else:
                    label = 0

                self.mri_dataset.append((mri_image_path, label))

    def __len__(self):
        return len(self.mri_dataset)

    def __getitem__(self, index):
        img_path, label = self.mri_dataset[index]

        # read image
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label
