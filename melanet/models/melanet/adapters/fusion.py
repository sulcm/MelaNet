import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LearnableAdapter
from .config import FeatureAdapterConfig


class FusionAdapter(LearnableAdapter):
    def __init__(
        self,
        dim_A: int,
        dim_B: int,
        fused_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        hidden_act: str = "gelu",
        input_l2_norm: bool = False,
        output_l2_norm: bool = True
    ):
        super(FusionAdapter, self).__init__()

        assert hidden_act in self.activation_str2fn.keys(), f"Entered unsupported name of activation function, must be one of {self.activation_str2fn.keys()}"
        self.__hidden_act = hidden_act

        self.proj_A = nn.Sequential(
            nn.Linear(dim_A, hidden_dim),
            self.activation_str2fn[self.__hidden_act](),
            nn.Dropout(dropout)
        )

        self.proj_B = nn.Sequential(
            nn.Linear(dim_B, hidden_dim),
            self.activation_str2fn[self.__hidden_act](),
            nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation_str2fn[self.__hidden_act](),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fused_dim)
        )

        self.input_l2_norm = input_l2_norm
        self.output_l2_norm = output_l2_norm

        self._is_fitted = False

    def forward(self, features_A: torch.Tensor, features_B: torch.Tensor) -> torch.Tensor:
        assert self.training or self._is_fitted, "FusionAdapter is not fitted. Call `fit()` first."

        if self.input_l2_norm:
            features_A = F.normalize(features_A, p=2, dim=-1)
            features_B = F.normalize(features_B, p=2, dim=-1)

        h_A = self.proj_A(features_A)
        h_B = self.proj_B(features_B)

        h = torch.cat([h_A, h_B], dim=1)
        z_fused = self.fusion(h)

        if self.output_l2_norm:
            z_fused = F.normalize(z_fused, p=2, dim=-1)
        return z_fused

    def save_as_pretrained(self, save_path: str, allow_overwrite: bool = True) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=allow_overwrite)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "dim_A": self.proj_A[0].in_features,
                    "dim_B": self.proj_B[0].in_features,
                    "hidden_dim": self.fusion[-1].in_features,
                    "fused_dim": self.fusion[-1].out_features,
                    "dropout": self.fusion[-2].p,
                    "hidden_act": self.__hidden_act,
                    "input_l2_norm": self.input_l2_norm,
                    "output_l2_norm": self.output_l2_norm
                }
            },
            save_path
        )

    @classmethod
    def from_config(cls, config: FeatureAdapterConfig) -> "FusionAdapter":
        assert config.in_features_A is not None and config.in_features_B is not None
        init_kwargs = {
            "dim_A": config.in_features_A,
            "dim_B": config.in_features_B,
            "fused_dim": config.out_features,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "hidden_act": config.hidden_act,
            "input_l2_norm": config.input_l2_norm,
            "output_l2_norm": config.output_l2_norm
        }
        init_kwargs = {k: v for k, v in init_kwargs if v is not None}
        return cls(**init_kwargs)