import shutil
import sys

import kaggle

from handwriting_recognition.utils import get_dataset_folder_path


def download_data_from_kaggle() -> None:
    dataset_folder = get_dataset_folder_path()
    output_dataset_dir = dataset_folder.joinpath("raw")

    print(f"Downloading raw data to {output_dataset_dir}")
    confirmation = input(
        "Type Y to confirm. This will overwrite existing files in this directory. Type anything else or leave blank to exit: "
    )

    if not confirmation.lower() == "y":
        sys.exit(0)

    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dataset_dir)

    dataset_reference = kaggle.api.dataset_list(search="handwriting-recognition", user="landlord")
    if not len(dataset_reference) == 1:
        raise NotImplementedError("Found multiple datasets... Check the script and fix the dataset reference")
    dataset_reference = dataset_reference[0].ref
    print(f"Deleted existing files under {output_dataset_dir} (if they existed).\nStarting download...")

    kaggle.api.dataset_download_files(dataset_reference, path=output_dataset_dir, unzip=True)

    for dataset in ["train", "test", "validation"]:
        original_path = output_dataset_dir / f"{dataset}_v2" / dataset
        new_path = output_dataset_dir / dataset
        shutil.move(original_path, new_path)
        shutil.rmtree(original_path.parent)
        shutil.move(output_dataset_dir / f"written_name_{dataset}_v2.csv", dataset_folder.joinpath(f"{dataset}.csv"))

    print("Finished!")


if __name__ == "__main__":
    sys.exit(download_data_from_kaggle())
