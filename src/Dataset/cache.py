# External Imports
import os
import json
from typing import Literal

# Internal Imports
from Typedef.Patients import Patients, RejectedSegments


def save_cache(data: Patients, path: os.PathLike, data_type: Literal["accepted", "rejected"] = "accepted" ):
    JSON_DATA = {}
    if data_type == "accepted":
        for patient_id, mri_segments in data.items():
            for mri_segment in mri_segments:
                mri_image_path, mri_mask_path = mri_segment
                if patient_id not in JSON_DATA:
                    JSON_DATA[patient_id] = []
                JSON_DATA[patient_id].append([str(mri_image_path), str(mri_mask_path)])

    if data_type == "rejected":
        data = {'patient_id': [], 'segment_id': [], 'reject_msg': []}

        # flatten the rejected segment dict to readable rows for pandas
        for patient_id, mri_segments in data.items():
            for mri_segment in mri_segments:
                if patient_id not in JSON_DATA:
                    JSON_DATA[patient_id] = []
                JSON_DATA[patient_id].append([str(mri_segment[0]), mri_segment[1]])

    JSON_DATA = json.dumps(JSON_DATA, indent=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'w') as f:
            f.write(JSON_DATA)
    except Exception as e:
        print(f"Error writing cache: {e}")
    print(f"[INFO] Cache saved to {path}")


def load_cache(path: os.PathLike) -> Patients | RejectedSegments:
    try:
        # Open the file in read mode ('r') with the 'with' statement
        with open(path, 'r') as f:
            # Use json.load() to parse the file data into a Python object
            data: Patients | RejectedSegments = json.load(f)

    except FileNotFoundError:
        print("Error: The file 'data.json' was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file. Check for invalid JSON content.")
    print(f"[INFO] Cache loaded from {path}")
    return data
