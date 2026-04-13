import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .learnable_base import LearnableAdapter
from .config import FeatureAdapterConfig


class LinearAdapter(LearnableAdapter):
    adapter_type = "linear"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        input_l2_norm: bool = False,
        output_l2_norm: bool = False
    ):
        super(LinearAdapter, self).__init__()

        self.projection = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )

        self.input_l2_norm = input_l2_norm
        self.output_l2_norm = output_l2_norm

        self._is_fitted = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        assert self.training or self._is_fitted, "LinearAdapter is not fitted. Call `fit()` first."

        if self.input_l2_norm:
            features = F.normalize(features, p=2, dim=-1)

        proj_features = self.projection(features)

        if self.output_l2_norm:
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
                    "input_l2_norm": self.input_l2_norm,
                    "output_l2_norm": self.output_l2_norm
                }
            },
            save_path
        )

    @classmethod
    def from_config(cls, config: FeatureAdapterConfig) -> "LinearAdapter":
        assert config.in_features is not None and isinstance(config.in_features, int)
        init_kwargs = {
            "in_features": config.in_features,
            "out_features": config.out_features,
            "bias": config.bias,
            "input_l2_norm": config.input_l2_norm,
            "output_l2_norm": config.output_l2_norm
        }
        init_kwargs = {k: v for k, v in init_kwargs if v is not None}
        return cls(**init_kwargs)