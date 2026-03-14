# External
import os
import sys
from pathlib import Path
import kagglehub

# Internal
from DataIntegrity import data_integrity_check
from Dataset.cache import save_cache, load_cache
from Dataset import split_patients

sys.path.insert(0, str(Path(__file__).parent.parent))

SPLIT_SEED = 42

def main():
    # Define data paths
    accepted_path = Path("data/processed/cache/accepted_data.json")
    rejected_path = Path("data/processed/cache/")



    # Download dataset
    data_path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    data_path = Path(data_path) / "kaggle_3m"
    print("Path to dataset files:", data_path)

    # DATA INTEGRITY CHECK
    accepted_data, rejected_data = data_integrity_check(data_path)

    # Cache the accepted data
    save_cache(accepted_data, accepted_path, data_type="accepted")
    save_cache(rejected_data, rejected_path, data_type="rejected")

   # train, valid, test = split_patients(accepted_data, SPLIT_SEED)

if __name__ == "__main__":
    main()