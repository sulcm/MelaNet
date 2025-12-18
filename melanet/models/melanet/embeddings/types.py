import torch
import numpy as np

from typing import Optional
from dataclasses import dataclass, field

from .utils import normalize_embeddings
from ..utils import tensor2numpy


@dataclass
class FeatureExtractorOutput:
    ft_embeddings: Optional[torch.Tensor] = field(default=None)
    zero_shot_embeddings: Optional[torch.Tensor] = field(default=None)

    def have_ft_embeddings(self) -> bool:
        return self.ft_embeddings is not None

    def have_zero_shot_embeddings(self) -> bool:
        return self.zero_shot_embeddings is not None

    def sum(self, alpha: float) -> torch.Tensor:
        if  self.have_ft_embeddings() and self.have_zero_shot_embeddings():
            return normalize_embeddings(
                alpha * self.ft_embeddings + (1.0 - alpha) * self.zero_shot_embeddings
            )
        elif self.have_ft_embeddings():
            return self.ft_embeddings
        elif self.have_zero_shot_embeddings():
            return self.zero_shot_embeddings
        else:
            raise ValueError("Output has no embeddings")

    def concat(self) -> torch.Tensor:
        if  self.have_ft_embeddings() and self.have_zero_shot_embeddings():
            return normalize_embeddings(
                torch.cat(
                    [self.ft_embeddings, self.zero_shot_embeddings],
                    dim=-1
                )
            )
        elif self.have_ft_embeddings():
            return self.ft_embeddings
        elif self.have_zero_shot_embeddings():
            return self.zero_shot_embeddings
        else:
            raise ValueError("Output has no embeddings")
    
    def to_dict(self) -> dict[str, Optional[np.ndarray]]:
        _obj_dict = {}
        _obj_dict["ft_embeddings"] = tensor2numpy(self.ft_embeddings) if self.have_ft_embeddings() else None
        _obj_dict["zero_shot_embeddings"] = tensor2numpy(self.zero_shot_embeddings) if self.have_zero_shot_embeddings() else None
        return _obj_dict