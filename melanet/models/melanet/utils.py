import torch
import numpy as np

from typing import Union
from numbers import Number
from pathlib import Path
from datetime import datetime, UTC


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


def kwargs2cli(**kwargs) -> str:
    cli_args = [
        f"{key_args}={value_arg}"
        for key_args, value_arg in kwargs.items()
    ]
    return ",".join(cli_args) if cli_args else ""


def handle_existing_path(path: Union[str, Path]) -> str:
    _path = Path(path)

    if _path.exists():
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{_path.name}"
        _path = _path.with_name(new_filename)

    return str(_path)