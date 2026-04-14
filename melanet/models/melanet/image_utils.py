# Implementation based on `transformers.image_utils`
# GitHub Link: https://github.com/huggingface/transformers/blob/main/src/transformers/image_utils.py

import torch
import numpy as np

from typing import Union

from PIL import Image
from torchvision.transforms.v2.functional import to_pil_image


ImageInput = Union[
    Image.Image, np.ndarray, torch.Tensor, list[Image.Image], list[np.ndarray], list[torch.Tensor]
]


def ensure_pil_image(inpt, mode=None) -> Image.Image:
    if isinstance(inpt, Image.Image):
        return inpt
    return to_pil_image(inpt, mode=mode)


def make_flat_list_of_images(
    images: Union[list[ImageInput], ImageInput],
    expected_ndims: int = 3,
) -> ImageInput:
    """
    Ensure that the output is a flat list of images. If the input is a single image, it is converted to a list of length 1.
    If the input is a nested list of images, it is converted to a flat list of images.
    Args:
        images (`Union[list[ImageInput], ImageInput]`):
            The input image.
        expected_ndims (`int`, *optional*, defaults to 3):
            The expected number of dimensions for a single input image.
    Returns:
        list: A list of images or a 4d array of images.
    """
    # If the input is a nested list of images, we flatten it
    if (
        isinstance(images, (list, tuple))
        and all(isinstance(images_i, (list, tuple)) for images_i in images)
        and all(is_valid_list_of_images(images_i) or not images_i for images_i in images)
    ):
        return [img for img_list in images for img in img_list]

    if isinstance(images, (list, tuple)) and is_valid_list_of_images(images):
        if is_pil_image(images[0]) or images[0].ndim == expected_ndims:
            return images
        if images[0].ndim == expected_ndims + 1:
            return [img for img_list in images for img in img_list]

    if is_valid_image(images):
        if is_pil_image(images) or images.ndim == expected_ndims:
            return [images]
        if images.ndim == expected_ndims + 1:
            return list(images)

    raise ValueError(f"Could not make a flat list of images from {images}")


def is_pil_image(img) -> bool:
    return isinstance(img, Image.Image)


def is_valid_image(img) -> bool:
    return is_pil_image(img) or isinstance(img, np.ndarray) or isinstance(img, torch.Tensor)


def is_valid_list_of_images(images: list) -> bool:
    return images and all(is_valid_image(image) for image in images)