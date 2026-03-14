# External Imports
import os
import json

# Internal Imports
from Typedef.Patients import Patients


def save_cache(data: Patients, path: os.PathLike):
    JSON_DATA = {}
    for patient_id, mri_segments in data.items():
        for mri_segment in mri_segments:
            mri_image_path, mri_mask_path = mri_segment
            if patient_id not in JSON_DATA:
                JSON_DATA[patient_id] = []
            JSON_DATA[patient_id].append([str(mri_image_path), str(mri_mask_path)])

    JSON_DATA = json.dumps(JSON_DATA, indent=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'w') as f:
            f.write(JSON_DATA)
    except Exception as e:
        print(f"Error writing cache: {e}")
    print(f"[INFO] Cache saved to {path}")


def load_cache(path: os.PathLike) -> Patients:
    pass
