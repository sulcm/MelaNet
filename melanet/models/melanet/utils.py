import torch
import numpy as np

from typing import Union
from numbers import Number


def tensor2numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def tensor2value(t: torch.Tensor) -> Union[Number, list]:
    return t.item() if t.numel() == 1 else t.tolist()


def resolve_device(device: Union[str, torch.device] = "cuda") -> torch.device:
    if isinstance(device, str):
        if device != "cpu":
            device = device if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
    elif isinstance(device, torch.device):
        device_type = device.type
        if device_type != "cpu":
            device_type = device_type if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
    else:
        pass
    return device