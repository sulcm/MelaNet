import torch
import torchvision.transforms.v2.functional as F_torch_augmentation

from typing import Optional, Callable
from functools import partial


def center_crop(
    inpt: torch.Tensor,
    area_perc: Optional[float] = 0.8
) -> torch.Tensor:
    assert 0.0 <= area_perc <= 1.0
    if not area_perc or area_perc == 1.0 or area_perc == 0.0:
        return inpt

    in_height, in_width = F_torch_augmentation.get_size(inpt)
    output_size = [
        int(in_height * area_perc),
        int(in_width * area_perc)
    ]
    return F_torch_augmentation.center_crop(
        inpt,
        output_size
    )


def build_view_transformations(
    center_crop_area_perc: Optional[float] = 0.8,
    rotate_angles: list[float] = None
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    if rotate_angles is None:
        rotate_angles = [15, 90, 270, 345]

    return {
        "center_crop": partial(center_crop, area_perc=center_crop_area_perc),
        "horizontal_flip": F_torch_augmentation.horizontal_flip,
        **{
            f"rotate_{angle}": partial(
                F_torch_augmentation.rotate,
                angle=angle,
            )
            for angle in rotate_angles
        },
    }