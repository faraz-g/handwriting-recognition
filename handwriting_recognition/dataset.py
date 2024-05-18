import os
from typing import Any

import cv2
import pandas as pd

import albumentations
from albumentations import Compose
from torch.utils.data import Dataset
from functools import cached_property
from transformers import ViTImageProcessor


def train_augmentations():
    augmentations = Compose(
        [
            albumentations.ImageCompression(quality_lower=35, quality_upper=100, p=0.1),
            albumentations.GaussNoise(p=0.1),
            albumentations.GaussianBlur(blur_limit=3, p=0.05),
        ]
    )
    return augmentations


class HandWritingDataset(Dataset):
    def __init__(
        self,
        data_path: os.PathLike,
        image_processor: ViTImageProcessor,
        augmentations: Compose | None = None,
    ) -> None:
        super().__init__()
        df = pd.read_csv(data_path)
        df = df.reset_index()

        self.df = df
        self.augmentations = augmentations
        self.image_processor = image_processor

    @property
    def max_length(self) -> int:
        return self.df["label"].apply(len).max()

    @cached_property
    def char_set(self) -> list[str]:
        out = set()
        for label in self.df["label"]:
            for ch in label:
                out.add(ch)

        return list(out)

    def __getitem__(self, index: int) -> Any:
        image_data = self.df.iloc[index]
        label = image_data["label"]
        image_path = image_data["file_path"]
        image = cv2.imread(image_path)

        if self.augmentations:
            image = self.augmentations(image=image)["image"]

        inputs = self.image_processor(image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(), label

    def __len__(self) -> int:
        return len(self.df)
