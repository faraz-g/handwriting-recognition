from pathlib import Path


def get_dataset_folder_path():
    return Path(__file__).parent.parent / "dataset"
