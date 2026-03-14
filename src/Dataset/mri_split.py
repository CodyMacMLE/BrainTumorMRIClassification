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

def tumor_ratio(data: Patients) -> float:
    total_segments = 0
    tumor_segments = 0
    for mri_segments in data.values():
        for segment in mri_segments:
            total_segments += 1
            _, mask_path = segment
            mask_img = Image.open(mask_path).convert('L')
            if np.array(mask_img).max() > 0:
                tumor_segments += 1

    return tumor_segments / total_segments

def within_allowance(tested_ratio: float, total_ratio: float, allowance: float) -> bool:
    return (total_ratio - allowance) < tested_ratio < (total_ratio + allowance)

def split_patients(data: Patients, seed: int = 42) -> tuple[Patients, Patients, Patients]:
    try:
        print(f"[INFO]  Splitting dataset with seed {seed}...")
        # Split at the patient level
        # Grab keys to split
        data_keys = list(data.keys())
        train_valid_keys, test_keys = train_test_split(data_keys, test_size = 0.15, random_state = seed)
        train_keys, valid_keys = train_test_split(train_valid_keys, test_size = 0.1765, random_state = seed)
        # reconstruct patient lists using keys
        train_data = reconstruct_list(data, train_keys)
        valid_data = reconstruct_list(data, valid_keys)
        test_data = reconstruct_list(data, test_keys)

        # Check if the tumor segment ratio across the split is within ±5% of your overall dataset ratio across all three splits
        total_tumor_ratio = tumor_ratio(data)
        train_tumor_ratio = tumor_ratio(train_data)
        valid_tumor_ratio = tumor_ratio(valid_data)
        test_tumor_ratio = tumor_ratio(test_data)

        allowance = total_tumor_ratio * 0.05

        if not within_allowance(train_tumor_ratio, total_tumor_ratio, allowance):
            raise AssertionError
        if not within_allowance(valid_tumor_ratio, total_tumor_ratio, allowance):
            raise AssertionError
        if not within_allowance(test_tumor_ratio, total_tumor_ratio, allowance):
            raise AssertionError

        return train_data, valid_data, test_data
    except AssertionError as e:
        raise AssertionError(f"[ERROR] Tumor segment ratio was not within ±5% of your overall dataset ratio with seed {seed}")