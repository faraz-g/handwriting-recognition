import os
from typing import Any

import cv2
import pandas as pd

# import torch
from albumentations import Compose

# from albumentations.pytorch.transforms import ToTensorV2
# from PIL import Image
from torch.utils.data import Dataset


class HandWritingDataset(Dataset):
    def __init__(
        self,
        data_path: os.PathLike,
        augmentations: Compose | None,
    ) -> None:
        super().__init__()
        df = pd.read_csv(data_path)
        df = df.reset_index()

        self.df = df
        self.augmentations = augmentations

    def __getitem__(self, index: int) -> Any:
        image_data = self.images_df.iloc[index]
        label = image_data["label"]
        image_path = image_data["file_path"]

        image = cv2.imread(image_path)

        # TODO Apply augmentations here
        # TODO Convert label to tensor
        # TODO Convert image to tensor

        return image, label

    def __len__(self) -> int:
        return len(self.images_df)
