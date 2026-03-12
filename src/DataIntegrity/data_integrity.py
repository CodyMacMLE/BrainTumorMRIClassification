# ./src/data_integrity.py

# External Imports
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import cv2 as cv
import os
from pathlib import Path

# Internal Imports
from typedef.Patients import Patients, MaskImage, MriImage, PatientId
from helpers.DataIntegrity import is_valid_mri, primarily_white_in_mask

def data_integrity_check(data_path: os.PathLike, rejected_path: os.PathLike | None = None, threshold: float = 0.75) -> tuple[Patients, dict[PatientId, list[tuple[int, str]]]]:
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
    rejected_segments: dict[PatientId, list[tuple[int, str]]] = {}

    # Loop through all patients
    for idx, patient_addr in enumerate(tqdm((glob.glob(f"{str(data_path)}/*")), desc = "[INFO] Data Integrity Checks", position = 0, dynamic_ncols = True)):
        # Store the patient id for use later
        patient_id: PatientId = patient_addr.split('/')[-1]

        # Loop through all the images for the patient
        for img_addr in glob.glob(f"{patient_addr}/*"):

            # Logic runs if the image is a mask
            if img_addr.endswith('_mask.tif'):
                segment_name = img_addr.split('/')[-1].replace("_mask.tif", "")
                segment_id = int(segment_name.split('_')[-1])
                # Store the mask as a ndarray
                mask_img: MaskImage = cv.imread(img_addr, cv.IMREAD_GRAYSCALE)

                # First Integrity Check: Bad image read (MASK)
                if mask_img is None:
                    patient_rejected_segments = rejected_segments.setdefault(patient_id, [])
                    patient_rejected_segments.append((segment_id, "[Error] Mask read was none type"))
                    # Break loop for this mask
                    continue

                # Second Integrity Check: An MRI Image pair does not exist
                mri_img: MriImage = cv.imread(f"{patient_addr}/{segment_name}.tif")
                if mri_img is None:
                    patient_rejected_segments = rejected_segments.setdefault(patient_id, [])
                    patient_rejected_segments.append((segment_id, "[Error] Mask image has no matcher MRI image pair"))
                    # Break loop for this mask
                    continue

                # Third Integrity Check: Images are the proper shape
                if not (mask_img.shape == (256,256) and mri_img.shape == (256,256,3)):
                    patient_rejected_segments = rejected_segments.setdefault(patient_id, [])
                    patient_rejected_segments.append((segment_id, "[Error] Mask and MRI image are not the right shape"))
                    # Break loop for this mask
                    continue

                # Fourth Integrity Check: Check if the MRI image is a valid image
                if not is_valid_mri(mri_img):
                    patient_rejected_segments = rejected_segments.setdefault(patient_id, [])
                    patient_rejected_segments.append((segment_id, "[Error] MRI image is not a valid image"))
                    # Break loop for this mask
                    continue

                # Fifth Integrity Check: Check if the mask is over 75% a mask (This means the tumor is not accurately zoned)
                if primarily_white_in_mask(mask_img, threshold): # the default threshold is 0.75
                    patient_rejected_segments = rejected_segments.setdefault(patient_id, [])
                    patient_rejected_segments.append((segment_id, "[Error] Mask image does not accurately zone tumor"))
                    # Break loop for this mask
                    continue

                # All checks pass append to patients
                patient_accepted_segments = patients.setdefault(patient_id, [])
                patient_accepted_segments.append((mri_img, mask_img))

    # if rejected path write to csv
    if rejected_path:
        print(f"[Info] Saving rejected segments to csv")
        data = { 'patient_id': [], 'segment_id': [], 'reject_msg': [] }

        # flatten the rejected segment dict to readable rows for pandas
        for patient, segments in rejected_segments.items():
            for segment in segments:
                data['patient_id'].append(patient)
                data['segment_id'].append(segment[0])
                data['reject_msg'].append(segment[1])

        # create dataframe and order by patient then segment
        df = pd.DataFrame(data)
        df = df.sort_values(by=['patient_id', 'segment_id'])

        # make directory if path does not exist and write to csv
        rejected_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(f"{str(rejected_path)}/output.csv")

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