# External Imports
import torch
import numpy as np
import cv2 as cv
from PIL import Image

# Internal Imports
from Typedef.Patients import Patients

class MriDataset(torch.utils.data.Dataset):
    def __init__(self, data: Patients, transform=None):
        """
        Initializes the MRI Dataset to use in training, testing, and validation
        :param data: The data of patients that have passed data integrity. The data needs to have been split
        prior to this step and given as the Patients datatype which stores the data as a cv2 image within a ndarray type (numpy).
        :param transform: I am unsure what this is
        """

        self.mri_images = []
        self.transform = transform

        # Iterate through patients dict
        for _ , mri_segments in data.items():
            # Iterate through the segments list
            for mri_segment in mri_segments:
                mri_image, mri_mask = mri_segment
                # create label
                if mri_mask.max() > 0:
                    label = 1
                else:
                    label = 0

                self.mri_images.append((mri_image, label))

    def __len__(self):
        return len(self.mri_images)

    def __getitem__(self, index):
        image, label = self.mri_images[index]

        # convert cv2 bgr to rgb
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # convert mri to PIL format
        image = Image.fromarray(image, 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
