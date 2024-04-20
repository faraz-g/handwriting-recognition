import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import kaggle
import numpy as np
import pandas as pd
from PIL import Image

# from skimage.measure import label
from tqdm import tqdm

from handwriting_recognition.utils import get_dataset_folder_path


def filter_missing_rows(labels: pd.DataFrame, dataset_folder: os.PathLike) -> pd.DataFrame:
    def _return_path_if_file_missing(file: str, directory: os.PathLike) -> str:
        if not os.path.exists(os.path.join(directory, file)):
            return file
        else:
            return None

    # Remove rows where there is no label
    labels = labels[~labels["IDENTITY"].isna()]
    labels = labels[~labels["I'M NOT SURE"].isna()]
    labels = labels[~labels["EMPTY"].isna()]

    # Remove rows where the filename is missing
    partial_func = partial(_return_path_if_file_missing, directory=dataset_folder)

    missing_files = set()
    file_paths = labels["FILENAME"].to_list()
    with ThreadPoolExecutor(2 * os.cpu_count()) as p:
        with tqdm(total=len(file_paths)) as pbar:
            for v in p.map(partial_func, file_paths, chunksize=4):
                pbar.update()
                missing_files.add(v)

    labels = labels[~labels["FILENAME"].isin(missing_files)]

    return labels


def process_single_image(image_path: os.PathLike, target_folder: os.PathLike) -> None:
    image = Image.open(image_path).convert("L")
    # Binarize image
    im_arr = np.array(image)
    im_arr = np.invert(im_arr).astype("float64")
    im_arr /= 255.0
    im_arr[im_arr < 0.1] = 0
    im_arr[im_arr != 0] = 1
    # Crop unwanted regions
    # labelled_arr = label(im_arr)

    # Save image
    processed_image = Image.fromarray(im_arr)
    out_path = Path(os.path.join(target_folder, os.path.basename(image_path)))
    processed_image.save(out_path.with_suffix(".tiff"))


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

    pre_processed_folder.mkdir()
    for dataset in ["train", "test", "validation"]:
        labels_file_name = f"{dataset}.csv"
        image_folder = dataset_folder / "raw" / dataset
        target_image_folder = pre_processed_folder / dataset

        process_images(image_folder=image_folder, target_folder=target_image_folder)

        labels = pd.read_csv(dataset_folder / "raw" / labels_file_name)
        labels = filter_missing_rows(labels=labels, dataset_folder=target_image_folder)
        labels = labels.rename(columns={"FILENAME": "file_name", "IDENTITY": "label"})
        labels["file_path"] = labels["file_name"].apply(lambda x: os.path.join(target_image_folder, x))
        labels["label"] = labels["label"].apply(lambda x: x.lower())

        labels.to_csv(pre_processed_folder / labels_file_name, index=None)


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
