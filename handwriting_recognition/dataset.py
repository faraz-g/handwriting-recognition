import os
from functools import cached_property
from typing import Any

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class HandWritingDataset(Dataset):
    def __init__(self, data_path: os.PathLike, img_size: int, size: int = None) -> None:
        super().__init__()
        df = pd.read_csv(data_path)
        df = df.reset_index()

        if size is None:
            self.df = df
        else:
            self.df = df[:size]

        self.augmentations = A.Compose(
            [
                A.LongestMaxSize(max_size=img_size, p=1, interpolation=cv2.INTER_NEAREST),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, value=1, border_mode=cv2.BORDER_CONSTANT),
                ToTensorV2(),
            ]
        )

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
        image = np.array(Image.open(image_path))
        image = image.astype(np.float32)

        # plt.imshow(image)
        # plt.show()
        image = self.augmentations(image=image, force_apply=True)["image"]

        # plt.imshow(image.swapaxes(0, 1).swapaxes(1, 2))
        # plt.show()

        return image, label

    def __len__(self) -> int:
        return len(self.df)
