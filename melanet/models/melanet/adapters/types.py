import torch
import numpy as np

from typing import Union
from dataclasses import dataclass

from transformers.utils.generic import ModelOutput


@dataclass
class AdapterOutput(ModelOutput):
    adapter_output: Union[torch.Tensor, np.ndarray]