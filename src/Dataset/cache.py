# External Imports
import os
import json
from typing import Literal
from pathlib import Path

# Internal Imports
from Typedef.Patients import Patients, RejectedSegments


def save_cache(data: Patients, save_path: os.PathLike, root_path: os.PathLike, data_type: Literal["accepted", "rejected"] = "accepted" ):
    JSON_DATA = {}
    if data_type == "accepted":
        for patient_id, mri_segments in data.items():
            for mri_segment in mri_segments:
                mri_image_path, mri_mask_path = mri_segment
                if patient_id not in JSON_DATA:
                    JSON_DATA[patient_id] = []
                JSON_DATA[patient_id].append([mri_image_path.relative_to(root_path), mri_mask_path.relative_to(root_path)])

    if data_type == "rejected":
        data = {'patient_id': [], 'segment_id': [], 'reject_msg': []}

        # flatten the rejected segment dict to readable rows for pandas
        for patient_id, mri_segments in data.items():
            for mri_segment in mri_segments:
                if patient_id not in JSON_DATA:
                    JSON_DATA[patient_id] = []
                JSON_DATA[patient_id].append([str(mri_segment[0]), mri_segment[1]])

    JSON_DATA = json.dumps(JSON_DATA, indent=4)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(save_path, 'w') as f:
            f.write(JSON_DATA)
    except Exception as e:
        print(f"Error writing cache: {e}")
    print(f"[INFO] Cache saved to {save_path}")


def load_cache(path: os.PathLike, root_path: os.PathLike) -> Patients | RejectedSegments:
    data = None
    try:
        # Open the file in read mode ('r') with the 'with' statement
        with open(path, 'r') as f:
            # Use json.load() to parse the file data into a Python object
            data = json.load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cache miss: {path}") from e
    except json.JSONDecodeError as e:
        raise Exception("Error: Failed to decode JSON from the file. Check for invalid JSON content.") from e

    dict_data = {}
    # prefix images with root_path
    for patient_id, segments in data.items():
        for segment in segments:
            mri_img, mask_img = segment

            mri_img = root_path / Path(mri_img)
            mask_img = root_path / Path(mask_img)

            accepted_patient = dict_data.setdefault(patient_id, [])
            accepted_patient.append((mri_img, mask_img))


    print(f"[INFO]  Cache loaded from {path}")
    return dict_data
