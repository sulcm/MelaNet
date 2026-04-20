import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .learnable_base import LearnableAdapter
from .config import FeatureAdapterConfig
from .types import AdapterOutput
from ..utils import handle_existing_path


class LinearAdapter(LearnableAdapter):
    adapter_type = "linear"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.0,
        input_norm: bool = False,
        input_l2_norm: bool = False,
        output_l2_norm: bool = False
    ):
        super(LinearAdapter, self).__init__()

        self.__in_features = in_features
        self.__out_features = out_features
        self.__bias = bias
        self.__dropout = dropout
        self.__input_norm = input_norm

        self.projection = nn.Sequential(
            nn.LayerNorm(in_features) if input_norm else nn.Identity(),
            nn.Dropout(dropout),
        )
        _linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        nn.init.trunc_normal_(_linear.weight, std=0.01)
        if bias:
            nn.init.zeros_(_linear.bias)
        self.projection.append(_linear)

        self.input_l2_norm = input_l2_norm
        self.output_l2_norm = output_l2_norm

        self._is_fitted = False

    def forward(self, features: torch.Tensor, **kwargs) -> AdapterOutput:
        if self.input_l2_norm:
            features = F.normalize(features, p=2, dim=-1)

        proj_features = self.projection(features)

        if self.output_l2_norm:
            proj_features = F.normalize(proj_features, p=2, dim=-1)
        return AdapterOutput(adapter_output=proj_features)

    def save_as_pretrained(self, save_path: str, allow_overwrite: bool = True) -> None:
        if not allow_overwrite:
            save_path = handle_existing_path(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "adapter_type": self.adapter_type,
                "config": {
                    "in_features": self.__in_features,
                    "out_features": self.__out_features,
                    "bias": self.__bias,
                    "dropout": self.__dropout,
                    "input_norm": self.__input_norm,
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
            "dropout": config.dropout,
            "input_norm": config.input_norm,
            "input_l2_norm": config.input_l2_norm,
            "output_l2_norm": config.output_l2_norm
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        return cls(**init_kwargs)