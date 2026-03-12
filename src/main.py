# External
import os
from pathlib import Path

# Internal
from DataIntegrity import data_integrity_check


def main():
    # DATA INTEGRITY
    data_path = Path("data/raw/lgg-mri-segmentation/kaggle_3m")
    rejected_path = Path("data/processed/data-integrity")
    accepted_data, rejected_data = data_integrity_check(data_path, rejected_path)

if __name__ == "__main__":
    main()