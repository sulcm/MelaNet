import torch

from torchvision.transforms.v2.functional import to_image, to_dtype

from typing import Callable, Any


def create_views(
    image: Any,
    transforms: list[Callable[[torch.Tensor], torch.Tensor]] = None
) -> list[torch.Tensor]:
    t_image = to_dtype(
        to_image(image),
        dtype=torch.float32,
        scale=True
    )
    views = [t_image,]
    if transforms is None:
        return views

    for transform in transforms:
        views.append(
            transform(t_image)
        )
    return views


def create_views_batched(
    images: list[Any],
    transforms: list[Callable[[torch.Tensor], torch.Tensor]] = None
) -> list[torch.Tensor]:
    tensor_images = [
        to_dtype(
            to_image(img),
            dtype=torch.float32,
            scale=True
        )
        for img in images
    ] # shape: [B, C, H, W]
    if transforms is None:
        return tensor_images

    views = []
    for t_im in tensor_images:
        views.extend([t_im] + [t(t_im) for t in transforms])
    return views