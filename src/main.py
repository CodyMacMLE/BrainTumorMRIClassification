# External
import os
from pathlib import Path

# Internal
from DataIntegrity import data_integrity_check
from Dataset import split_patients

SPLIT_SEED = 42


def main():
    # DATA INTEGRITY
    data_path = Path("data/raw/lgg-mri-segmentation/kaggle_3m")
    rejected_path = Path("data/processed/data-integrity")
    accepted_data, rejected_data = data_integrity_check(data_path, rejected_path)

    train, valid, test = split_patients(accepted_data, SPLIT_SEED)

if __name__ == "__main__":
    main()