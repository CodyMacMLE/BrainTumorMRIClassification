# External Imports
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

# Internal Imports
from Typedef.Patients import Patients

def reconstruct_list(data: Patients, keys: list[str]) -> Patients:
    new_list: Patients = {}
    for key in keys:
        new_list[key] = data[key]

    return new_list

def patient_tumor_ratio(data) -> float:
    tumor_detected = 0
    total_segments = 0
    # Iterate through patients' mri segments
    for segment in data:
        total_segments += 1
        # Separate segment tuple (mri_image is irrelevant in this logic)
        _, mask_path = segment
        # Read image as grayscale
        mask_img = Image.open(mask_path).convert('L')
        # convert to np array and check if any pixel value is greater than 0 (indicating tumor presence)
        if np.array(mask_img).max() > 0:
            tumor_detected += 1

    # Calculate tumor ratio and assign to bin
    return tumor_detected / total_segments

def tumor_ratio_bins(data: Patients) -> list[int]:
    bins = [] # Current bins [ 0-20%, 21-40%, 41-100% ]
    # Iterate through patients
    for mri_segments in data.values():
        segment_tumor_ratio = patient_tumor_ratio(mri_segments)

        if 0 <= segment_tumor_ratio <= 0.2:
            bins.append(0)
        elif 0.2 < segment_tumor_ratio <= 0.4:
            bins.append(1)
        elif 0.4 < segment_tumor_ratio <= 1.0:
            bins.append(2)

    return bins

def within_allowance(tested_ratio: float, total_ratio: float, allowance: float) -> bool:
    return (total_ratio - allowance) < tested_ratio < (total_ratio + allowance)

def split_patients(data: Patients, seed: int = 42) -> tuple[Patients, Patients, Patients]:
    print(f"[INFO]  Splitting dataset with seed {seed}...")
    # Split at the patient level

    # Gets the patients id, and bins them according to their tumor ratio for stratification
    data_keys = list(data.keys())
    stratify_bins = tumor_ratio_bins(data)
    # pairs the key/bins to filter out after primary split
    key_bin_dict = dict(zip(data_keys, stratify_bins))

    # Primary split for test set with stratification
    train_valid_keys, test_keys = train_test_split(data_keys, test_size = 0.15, random_state = seed, stratify = stratify_bins)

    # Filter out used patients from the stratification bins for the secondary split
    stratify_bins = [key_bin_dict[key] for key in train_valid_keys]

    # Secondary split for train and valid sets with stratification
    train_keys, valid_keys = train_test_split(train_valid_keys, test_size = 0.1765, random_state = seed, stratify = stratify_bins)

    # reconstruct patient lists using keys
    train_data = reconstruct_list(data, train_keys)
    valid_data = reconstruct_list(data, valid_keys)
    test_data = reconstruct_list(data, test_keys)

    train_tumor_ratio = 0
    for mri_segments in train_data.values():
        train_tumor_ratio += patient_tumor_ratio(mri_segments)
    train_tumor_ratio = train_tumor_ratio / len(train_data)

    valid_tumor_ratio = 0
    for mri_segments in valid_data.values():
        valid_tumor_ratio += patient_tumor_ratio(mri_segments)
    valid_tumor_ratio = valid_tumor_ratio / len(valid_data)

    test_tumor_ratio = 0
    for mri_segments in test_data.values():
        test_tumor_ratio += patient_tumor_ratio(mri_segments)
    test_tumor_ratio = test_tumor_ratio / len(test_data)

    # print results
    print(f"[INFO]  Train Set: {len(train_data)}  | Tumor Ratio: {train_tumor_ratio:.3f}")
    print(f"[INFO]  Valid Set: {len(valid_data)}  | Tumor Ratio: {valid_tumor_ratio:.3f}")
    print(f"[INFO]  Test Set:  {len(test_data)}  | Tumor Ratio: {test_tumor_ratio:.3f}")

    return train_data, valid_data, test_data