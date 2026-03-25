# External Imports
import torch
from PIL import Image
import numpy as np

# Internal Imports
from Typedef.Patients import Patients
from .transforms import Transforms

class MriDataset(torch.utils.data.Dataset):
    def __init__(self, data: Patients, set_type=None, segmentation=False):
        """
        Initializes the MRI Dataset to use in training, testing, and validation
        :param data: The data of patients that have passed data integrity. The data needs to have been split
        prior to this step and given as the Patients datatype.
        :param set_type:
        :param segmentation:
        """

        self.mri_dataset = []
        self.segmentation = segmentation
        if set_type == "train":
            self.transforms = Transforms(pixel_transforms=True)
        if set_type == "test" or set_type == "validation":
            self.transforms = Transforms()

        for patient, segments in data.items():
            for segment in segments:
                if not segmentation:
                    mri, mask = segment
                    mask = Image.open(mask).convert('L')
                    if np.array(mask).max() > 0:
                        label = 1
                    else:
                        label = 0
                    segment = (mri, label)

                self.mri_dataset.append(segment)


    def __len__(self):
        return len(self.mri_dataset)


    def __getitem__(self, index):
        img_path, label = self.mri_dataset[index]

        # read image
        image = Image.open(img_path)

        if self.segmentation:
            label = Image.open(label).convert('L')
            label = np.array(label)

            label = (label > 0).astype(np.float32)

            image, label = self.transforms(image, label)

            label = torch.unsqueeze(label, 0)
        else:
            image = self.transforms(image)

        return image, label
