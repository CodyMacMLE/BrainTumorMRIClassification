# External
import os
import sys
from pathlib import Path
import shutil
import kagglehub

# Internal
sys.path.insert(0, str(Path(__file__).parent.parent))
from DataIntegrity import data_integrity_check
from Dataset.cache import save_cache, load_cache
from Typedef.Patients import Patients, RejectedSegments
from Dataset import split_patients

SPLIT_SEED = 43

# HELPERS
def download_dataset(download_path: os.PathLike) -> os.PathLike:
    if Path(download_path).exists():
        print(f"[INFO] Dataset already exists at {download_path}. Skipping download.")
        return Path(download_path, "kaggle_3m")

    data_path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    shutil.copytree(data_path, download_path, dirs_exist_ok=True)
    data_path = Path(download_path, "kaggle_3m")

    print("Path to dataset files:", data_path)
    return data_path


def build_cache(download_path, accepted_path, rejected_path):
    # If cache doesn't exist, download dataset and perform integrity check
    data_path = download_dataset(download_path)
    accepted_data, rejected_data = data_integrity_check(data_path)
    # Cache the accepted data
    save_cache(accepted_data, accepted_path, data_type="accepted")
    save_cache(rejected_data, rejected_path, data_type="rejected")

    return accepted_data, rejected_data


# MAIN
def main():
    # Define data paths
    download_path = Path("data/raw/lgg-mri-segmentation")
    accepted_path = Path("data/processed/cache/accepted_data.json")
    rejected_path = Path("data/processed/cache/rejected_data.json")

    accepted_data: Patients = {}
    rejected_data: RejectedSegments = {}
    # DATA INTEGRITY CHECK
    if accepted_path.exists():
        try:
            accepted_data = load_cache(accepted_path)
            rejected_data = load_cache(rejected_path)
        except Exception as e:
            print(f"[Error] Loading data cache: {e}")
            print("[INFO] Cache is corrupted or unreadable. Redownloading dataset and performing integrity check.")
            accepted_data, rejected_data = build_cache(download_path, accepted_path, rejected_path)
    else:
        accepted_data, rejected_data = build_cache(download_path, accepted_path, rejected_path)

    train, valid, test = split_patients(accepted_data, SPLIT_SEED)

    debugger = 0

if __name__ == "__main__":
    main()