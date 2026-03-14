# ./src/data_integrity.py

# External Imports
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
from PIL import Image
import os

# Internal Imports
from Typedef.Patients import Patients, RejectedSegments, MaskImage, MriImage, PatientId

# HELPERS
def primarily_white_in_mask(mask: np.ndarray, threshold: float = 0.75) -> bool:
    """
    Function checks if a grayscale mask is primarily white based on a threshold.
    :param mask: The np.ndarray that represents the mask to check. Expected in grayscale format (2D).
    :param threshold: The percentage of pixels required to be white, as a float (0.0 to 1.0).
    :return: True if white pixel ratio meets or exceeds the threshold, False otherwise.
    """
    if not (0 < threshold <= 1):
        raise ValueError("Threshold value must be lower than or equal to 1.0")
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D grayscale array. Convert to grayscale before passing in")

    white_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    return (white_pixels / total_pixels) >= threshold

def is_valid_mri(image: np.ndarray) -> bool:
    """
    Checks if the pixel values are a colored image and not a mask or blank image
    :param image:
    :return: True if valid mri image false if not
    """
    return not (np.all((image == 0) | (image == 255)))

# Main Function
def data_integrity_check(data_path: os.PathLike, threshold: float = 0.75) -> tuple[Patients, RejectedSegments]:
    """
    Checks the integrity of the data. Checks if images are read properly, the mri segment has a paired mask image;
    the images have the proper shape and channels, checks if the mri image is a valid colored image, and checks if the
    mask does not capture over a threshold amount.
    :param data_path: The path to the data holding the patients folder
    :param rejected_path: An optional path to store the rejected segments as a csv
    :param threshold: a float value between 0-1, that is the percentage of white the mask can detect
    :return: a tuple of dictionaries; Index [0] being the data that passed the test, Index [1] being the data that failed
    """
    patients: Patients = {}
    rejected_segments: RejectedSegments = {}

    # Loop through all patients
    for idx, patient_addr in enumerate(tqdm((glob.glob(f"{str(data_path)}/*")), desc = "[INFO] Data Integrity Checks", position = 0, dynamic_ncols = True)):
        # Store the patient id for use later
        patient_id: PatientId = os.path.basename(patient_addr)

        # Loop through all the images for the patient
        for img_addr in glob.glob(f"{patient_addr}/*"):

            # Logic runs if the image is a mask
            if img_addr.endswith('_mask.tif'):
                segment_name = os.path.basename(img_addr).replace("_mask.tif", "")
                mri_path = Path(patient_addr, f"{segment_name}.tif")
                # Store the mask as a PIL image
                try:
                    mask_img = Image.open(img_addr).convert('L')
                    mask_np_array = np.array(mask_img)
                except Exception:
                    # First Integrity Check: Bad image read (MASK)
                    patient_rejected_segments = rejected_segments.setdefault(patient_id, [])
                    patient_rejected_segments.append((Path(img_addr), "[Error] Mask read was none type"))
                    # Break loop for this mask
                    continue

                # Second Integrity Check: An MRI Image pair does not exist
                try:
                    mri_img = Image.open(mri_path)
                    mri_np_array = np.array(mri_img)
                except Exception:
                    patient_rejected_segments = rejected_segments.setdefault(patient_id, [])
                    patient_rejected_segments.append((mri_path, "[Error] Mask image has no matcher MRI image pair"))
                    # Break loop for this mask
                    continue

                # Third Integrity Check: Images are the proper shape
                if not (mask_np_array.shape == (256,256)):
                    patient_rejected_segments = rejected_segments.setdefault(patient_id, [])
                    patient_rejected_segments.append((Path(img_addr), "[Error] Mask and MRI image are not the right shape"))
                    # Break loop for this mask
                    continue
                if not (mri_np_array.shape == (256,256,3)):
                    patient_rejected_segments = rejected_segments.setdefault(patient_id, [])
                    patient_rejected_segments.append((mri_path, "[Error] Mask and MRI image are not the right shape"))
                    # Break loop for this mask
                    continue

                # Fourth Integrity Check: Check if the MRI image is a valid image
                if not is_valid_mri(mri_np_array):
                    patient_rejected_segments = rejected_segments.setdefault(patient_id, [])
                    patient_rejected_segments.append((mri_path, "[Error] MRI image is not a valid image"))
                    # Break loop for this mask
                    continue

                # Fifth Integrity Check: Check if the mask is over 75% a mask (This means the tumor is not accurately zoned)
                if primarily_white_in_mask(mask_np_array, threshold): # the default threshold is 0.75
                    patient_rejected_segments = rejected_segments.setdefault(patient_id, [])
                    patient_rejected_segments.append((Path(img_addr), "[Error] Mask image does not accurately zone tumor"))
                    # Break loop for this mask
                    continue

                # All checks pass append to patients
                mri_path = Path(f"{patient_addr}/{segment_name}.tif")
                mask_path = Path(img_addr)
                patient_accepted_segments = patients.setdefault(patient_id, [])
                patient_accepted_segments.append((mri_path, mask_path))

    # Print info to terminal
    accepted_len = 0
    for patient in patients.values():
        accepted_len += len(patient)

    rejected_len = 0
    for patient in rejected_segments.values():
        rejected_len += len(patient)

    print(f"[Info] Data Integrity Pipeline finished")
    print(f"[Info] Segments Accepted = {accepted_len} | Segments Rejected = {rejected_len}")

    # Return tuple
    return patients, rejected_segments