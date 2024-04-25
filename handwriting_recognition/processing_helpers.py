import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image


def get_image_boundary(im_arr: np.ndarray, direction: Literal["hor", "vert"]) -> tuple[int, int]:
    if direction not in ["hor", "vert"]:
        raise NotImplementedError

    axis = 0 if direction == "hor" else 1
    axis_length = im_arr.shape[1] if direction == "hor" else im_arr.shape[0]

    cols_with_zeros = np.any(~im_arr, axis=axis)

    left_bot_pixel = np.argmax(cols_with_zeros)
    right_top_pixel = len(cols_with_zeros) - np.argmax(cols_with_zeros[::-1])

    buffer = np.round(axis_length * 0.02).astype(np.int8)

    left_bot_pixel = left_bot_pixel - buffer if left_bot_pixel - buffer >= 0 else left_bot_pixel

    right_top_pixel = right_top_pixel + buffer if right_top_pixel + right_top_pixel < axis_length else right_top_pixel

    return left_bot_pixel, right_top_pixel


def resize_image(im_arr: np.ndarray, height: int, width: int) -> np.ndarray:
    im_arr_uint8 = im_arr.astype(np.uint8) * 255

    resized_im_arr_uint8 = cv2.resize(im_arr_uint8, (width, height), interpolation=cv2.INTER_NEAREST)

    # Convert back to bool type
    resized_im_arr_bool = (resized_im_arr_uint8 > 0).astype(bool)

    return resized_im_arr_bool


def process_single_image(image_path: os.PathLike, target_folder: os.PathLike) -> None:
    image = Image.open(image_path).convert("L")
    # Binarize image
    im_arr = np.array(image)
    # im_arr = np.invert(im_arr)
    im_arr = im_arr.astype("float64")
    im_arr /= 255.0
    im_arr[im_arr < 0.95] = 0
    im_arr[im_arr != 0] = 1

    # Crop unwanted regions
    im_arr = im_arr.astype("bool")

    left_boundary, right_boundary = get_image_boundary(im_arr=im_arr, direction="hor")

    bottom_boundary, top_boundary = get_image_boundary(im_arr=im_arr, direction="vert")

    im_arr = im_arr[bottom_boundary:top_boundary, left_boundary:right_boundary]

    # Resize
    im_arr = resize_image(im_arr=im_arr, height=64, width=256)

    # Save image
    processed_image = Image.fromarray(im_arr)
    out_path = Path(os.path.join(target_folder, os.path.basename(image_path)))
    processed_image.save(out_path.with_suffix(".tiff"))
