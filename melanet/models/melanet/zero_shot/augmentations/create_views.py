import torch

from torchvision.transforms.v2.functional import to_image, to_dtype

from typing import Callable, Any


def create_views(
    image: Any,
    transforms: list[Callable[[torch.Tensor], torch.Tensor]] = None,
    keep_original: bool = True
) -> list[torch.Tensor]:
    t_image = to_dtype(
        to_image(image),
        dtype=torch.float32,
        scale=True
    )
    if transforms is None:
        return [t_image,]
    
    views = []
    if keep_original:
        views.append(t_image)
    for transform in transforms:
        views.append(
            transform(t_image)
        )
    return views


def create_views_batched(
    images: list[Any],
    transforms: list[Callable[[torch.Tensor], torch.Tensor]] = None,
    keep_original: bool = True
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
    if keep_original:
        for t_im in tensor_images:
            views.extend([t_im] + [t(t_im) for t in transforms])
    else:
        for t_im in tensor_images:
            views.extend([t(t_im) for t in transforms])
    return views