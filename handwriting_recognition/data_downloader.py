import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import kaggle
import pandas as pd
from tqdm import tqdm

from handwriting_recognition.processing_helpers import process_single_image
from handwriting_recognition.utils import get_dataset_folder_path
import string


def filter_missing_rows(labels: pd.DataFrame, target_folder: os.PathLike) -> pd.DataFrame:
    def _return_path_if_file_missing(file: str, directory: os.PathLike) -> str:
        if not os.path.exists(os.path.join(directory, file)):
            return file
        else:
            return None

    # Remove rows where there is no label
    labels = labels[~labels["IDENTITY"].isna()]
    labels = labels[~labels["IDENTITY"].str.lower().isin(["i'm not sure", "unreadable", "empty"])]

    # Remove short and long labels
    labels = labels[labels["IDENTITY"].apply(len) > 3]
    labels = labels[labels["IDENTITY"].apply(len) < 20]

    # Remove rows where the filename is missing
    partial_func = partial(_return_path_if_file_missing, directory=target_folder)

    missing_files = set()
    labels["FILENAME"] = labels["FILENAME"].apply(lambda x: str(Path(x).with_suffix(".tiff")))
    file_paths = labels["FILENAME"].to_list()
    with ThreadPoolExecutor(2 * os.cpu_count()) as p:
        with tqdm(total=len(file_paths)) as pbar:
            for v in p.map(partial_func, file_paths, chunksize=4):
                pbar.update()
                missing_files.add(v)

    labels = labels[~labels["FILENAME"].isin(missing_files)]

    return labels


def clean_and_validate_labels(labels: pd.DataFrame, target_folder: str) -> pd.DataFrame:
    def _process_characters(in_str: str) -> str:
        out_str = ""
        for char in in_str:
            if char in string.ascii_lowercase + " ":
                out_str += char
            else:
                out_str += " "
        return out_str

    labels = labels.rename(columns={"FILENAME": "file_name", "IDENTITY": "label"})
    labels["file_path"] = labels["file_name"].apply(lambda x: os.path.join(target_folder, x))
    labels["label"] = labels["label"].apply(lambda x: x.lower())

    labels["label"] = labels["label"].apply(_process_characters)

    return labels


def process_images(image_folder: os.PathLike, target_folder: os.PathLike) -> None:
    os.makedirs(target_folder)
    all_files = [os.path.join(image_folder, file_path) for file_path in os.listdir(image_folder)]
    partial_func = partial(process_single_image, target_folder=target_folder)

    with ProcessPoolExecutor(os.cpu_count() - 1) as p:
        with tqdm(total=len(all_files)) as pbar:
            for _ in p.map(partial_func, all_files, chunksize=4):
                pbar.update()


def pre_process() -> None:
    dataset_folder = get_dataset_folder_path()
    pre_processed_folder = dataset_folder / "pre_processed"
    try:
        shutil.rmtree(pre_processed_folder)
    except FileNotFoundError:
        pass

    if not os.path.exists(dataset_folder / "raw"):
        raise NotImplementedError("You must download the data first before pre-processing!")

    pre_processed_folder.mkdir()
    for dataset in ["train", "test", "validation"]:
        print(f"Pre-processing {dataset} dataset")
        labels_file_name = f"{dataset}.csv"
        image_folder = dataset_folder / "raw" / dataset
        target_image_folder = pre_processed_folder / dataset

        process_images(image_folder=image_folder, target_folder=target_image_folder)

        print(f"Processing labels for {dataset} dataset")
        labels = pd.read_csv(dataset_folder / "raw" / labels_file_name)
        original_len = len(labels)
        labels = filter_missing_rows(labels=labels, target_folder=target_image_folder)
        labels = clean_and_validate_labels(labels, target_folder=target_image_folder)
        labels.to_csv(pre_processed_folder / labels_file_name, index=None)
        final_len = len(labels)

        print(
            f"Finished processing {dataset} labels! Removed: {original_len - final_len} bad labels. Original count: {original_len}. Final count: {final_len}"
        )


def download_data_from_kaggle() -> None:
    dataset_folder = get_dataset_folder_path()
    output_dataset_dir = dataset_folder.joinpath("raw")

    print(f"Downloading raw data to {output_dataset_dir}")
    confirmation = input(
        "Type Y to confirm. This will overwrite existing files in this directory. Type anything else or leave blank to exit: "
    )

    if not confirmation.lower() == "y":
        sys.exit(0)

    try:
        shutil.rmtree(output_dataset_dir)
    except FileNotFoundError:
        pass

    output_dataset_dir.mkdir(parents=True)

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
        shutil.move(
            output_dataset_dir / f"written_name_{dataset}_v2.csv", output_dataset_dir.joinpath(f"{dataset}.csv")
        )

    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--pre_process", action="store_true")

    args = parser.parse_args()

    if not (args.download or args.pre_process):
        raise NotImplementedError("You must either pass in --download, --pre_process or both")

    if args.download:
        download_data_from_kaggle()

    if args.pre_process:
        pre_process()

    sys.exit()
