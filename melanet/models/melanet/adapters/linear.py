import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LearnableAdapter


class LinearAdapter(LearnableAdapter):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        normalize: bool = False
    ):
        super(LinearAdapter, self).__init__()

        self.projection = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.normalize = normalize
        self._is_fitted = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        assert self.training or self._is_fitted, "LinearAdapter is not fitted. Call `fit()` first."

        proj_features = self.projection(features)
        if self.normalize:
            proj_features = F.normalize(proj_features, p=2, dim=-1)
        return proj_features

    def save_as_pretrained(self, save_path: str, allow_overwrite: bool = True) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=allow_overwrite)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "in_features": self.projection.in_features,
                    "out_features": self.projection.out_features,
                    "bias": self.projection.bias is not None,
                    "normalize": self.normalize
                }
            },
            save_path
        )