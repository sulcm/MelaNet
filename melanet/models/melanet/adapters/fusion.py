import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LearnableAdapter


class FusionAdapter(LearnableAdapter):
    def __init__(
        self,
        dim_A: int = 2048,
        dim_B: int = 768,
        hidden_dim: int = 1024,
        fused_dim: int = 512,
        dropout: float = 0.3,
        input_l2_norm: bool = False,
        output_l2_norm: bool = False
    ):
        super(FusionAdapter, self).__init__()

        self.proj_A = nn.Sequential(
            nn.Linear(dim_A, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.proj_B = nn.Sequential(
            nn.Linear(dim_B, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
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
                    "input_l2_norm": self.input_l2_norm,
                    "output_l2_norm": self.output_l2_norm
                }
            },
            save_path
        )