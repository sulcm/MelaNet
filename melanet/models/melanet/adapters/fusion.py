import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LearnableAdapter
from .config import FeatureAdapterConfig


class FusionAdapter(LearnableAdapter):
    def __init__(
        self,
        input_dims: list[int],
        fused_dim: int,
        use_attn: bool = False,
        attn_num_heads: int = 8,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        hidden_act: str = "gelu",
        input_l2_norm: bool = False,
        output_l2_norm: bool = False
    ):
        super(FusionAdapter, self).__init__()

        assert hidden_act in self.activation_str2fn.keys(), f"Entered unsupported name of activation function, must be one of {self.activation_str2fn.keys()}"
        self.__use_attn = use_attn
        self.__attn_num_heads = attn_num_heads
        self.__hidden_act = hidden_act
        self.__input_dims = input_dims
        self.__hidden_dim = hidden_dim
        self.__fused_dim = fused_dim
        self.__dropout = dropout
        self.__num_inputs = len(input_dims)

        self.input_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(i_dim, hidden_dim),
                self.activation_str2fn[self.__hidden_act](),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for i_dim in input_dims
        ])

        if self.__use_attn:
            self.input_pos_embed = nn.Parameter(
                torch.randn(1, self.__num_inputs, hidden_dim)
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)
            self.attn_fusion = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attn_num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attn_pool = nn.Linear(hidden_dim, 1)
        else:
            self.concat_fusion = nn.Sequential(
                nn.Linear(hidden_dim * self.__num_inputs, hidden_dim * 2),
                self.activation_str2fn[self.__hidden_act](),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation_str2fn[self.__hidden_act](),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fused_dim),
            nn.LayerNorm(fused_dim)
        )

        self.input_l2_norm = input_l2_norm
        self.output_l2_norm = output_l2_norm

        self._is_fitted = False

    def forward(self, input_features: list[torch.Tensor]) -> torch.Tensor:
        assert self.training or self._is_fitted, "FusionAdapter is not fitted. Call `fit()` first."
        assert len(input_features) == self.__num_inputs, f"Must passed same number of input as is initialized number of input projections, {len(input_features)} != {self.__num_inputs}"

        # Input projections into common hidden dim
        projected_inputs = []
        for features, input_projection in zip(input_features, self.input_projections):
            if self.input_l2_norm:
                features = F.normalize(features, p=2, dim=-1)
            h_input = input_projection(features)
            projected_inputs.append(h_input)

        # Fuse projected input into hidden dim
        H = torch.stack(projected_inputs, dim=1) # (B, N, D)
        if self.__use_attn:
            H = H + self.input_pos_embed # Add positional embeddings of inputs
            H_norm = self.attn_norm(H)
            H_attn, _ = self.attn_fusion(H_norm, H_norm, H_norm)
            H = H + H_attn
            # Pool fused features
            pool_weights = F.softmax(
                self.attn_pool(H),
                dim=1
            ) # (B, N, 1)
            h_fused = (H * pool_weights).sum(dim=1)
        else:
            H_flat = H.reshape(H.shape[0], -1) # (B, N * D)
            H_concat = self.concat_fusion(H_flat)
            H_residual = H.mean(dim=1)
            h_fused = H_concat + H_residual

        # Final (output) projection into desired dim
        z_fused = self.final_projection(h_fused)
        if self.output_l2_norm:
            z_fused = F.normalize(z_fused, p=2, dim=-1)

        return z_fused

    def save_as_pretrained(self, save_path: str, allow_overwrite: bool = True) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=allow_overwrite)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dims": self.__input_dims,
                    "hidden_dim": self.__hidden_dim,
                    "fused_dim": self.__fused_dim,
                    "dropout": self.__dropout,
                    "hidden_act": self.__hidden_act,
                    "use_attn": self.__use_attn,
                    "attn_num_heads": self.__attn_num_heads,
                    "input_l2_norm": self.input_l2_norm,
                    "output_l2_norm": self.output_l2_norm
                }
            },
            save_path
        )

    @classmethod
    def from_config(cls, config: FeatureAdapterConfig) -> "FusionAdapter":
        assert config.in_features is not None and isinstance(config.in_features, list)
        init_kwargs = {
            "input_dims": config.in_features,
            "hidden_dim": config.hidden_dim,
            "fused_dim": config.out_features,
            "dropout": config.dropout,
            "hidden_act": config.hidden_act,
            "use_attn": config.use_attn,
            "attn_num_heads": config.attn_num_heads,
            "input_l2_norm": config.input_l2_norm,
            "output_l2_norm": config.output_l2_norm
        }
        init_kwargs = {k: v for k, v in init_kwargs if v is not None}
        return cls(**init_kwargs)