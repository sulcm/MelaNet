import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.stochastic_depth import StochasticDepth

from .learnable_base import LearnableAdapter
from .config import FeatureAdapterConfig
from .types import AdapterOutput
from ..utils import handle_existing_path


class FusionAdapter(LearnableAdapter):
    adapter_type = "fusion"

    def __init__(
        self,
        input_dims: list[int],
        fused_dim: int,
        use_attn: bool = False,
        attn_num_heads: int = 2,
        attn_dropout: float = 0.0,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        modality_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_mode: str = "row",
        hidden_act: str = "gelu",
        input_norm: bool = False,
        input_l2_norm: bool = False,
        output_l2_norm: bool = False
    ):
        super(FusionAdapter, self).__init__()

        assert hidden_act in self.activation_str2fn.keys(), f"Entered unsupported name of activation function, must be one of {self.activation_str2fn.keys()}"
        self.__input_dims = input_dims
        self.__num_inputs = len(input_dims)
        self.__use_attn = use_attn
        self.__attn_num_heads = attn_num_heads
        self.__attn_dropout = attn_dropout
        self.__hidden_act = hidden_act
        self.__hidden_dim = hidden_dim
        self.__fused_dim = fused_dim
        self.__dropout = dropout
        self.__input_norm = input_norm
        self.__modality_dropout = modality_dropout
        self.__drop_path_rate = drop_path_rate
        self.__drop_path_mode = drop_path_mode

        _act_fcn = self.activation_str2fn[self.__hidden_act]

        self.input_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(i_dim) if input_norm else nn.Identity(),
                nn.Linear(i_dim, hidden_dim)
            )
            for i_dim in input_dims
        ])

        if self.__use_attn:
            self.input_pos_embed = nn.Parameter( # Modality (feature encoder) embeddings
                torch.randn(1, self.__num_inputs, hidden_dim)
            )
            nn.init.trunc_normal_(self.input_pos_embed, std=0.02)

            self.fusion_query = nn.Parameter( # CLS token
                torch.randn(1, 1, hidden_dim)
            )
            nn.init.normal_(self.fusion_query, std=1e-6)

            self.q_norm = nn.LayerNorm(hidden_dim)
            self.kv_norm = nn.LayerNorm(hidden_dim)
            self.attn_fusion = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attn_num_heads,
                dropout=attn_dropout,
                batch_first=True
            )
            self.post_attn_dropout = StochasticDepth(drop_path_rate, mode=drop_path_mode)
            self.post_ffn_dropout = StochasticDepth(drop_path_rate, mode=drop_path_mode)
            self.attn_ffn_norm = nn.LayerNorm(hidden_dim)
            self.attn_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                _act_fcn(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
        else:
            self.concat_drop_path = StochasticDepth(drop_path_rate, mode=drop_path_mode)
            self.concat_proj = nn.Linear(hidden_dim * self.__num_inputs, hidden_dim)
            self.concat_gate = nn.Sequential(
                nn.Linear(hidden_dim * self.__num_inputs, hidden_dim),
                nn.Sigmoid()
            )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_drop_path = StochasticDepth(drop_path_rate, mode=drop_path_mode)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _act_fcn(),
            nn.Dropout(dropout)
        )
        self.output_layer = nn.Linear(hidden_dim, fused_dim)

        self.input_l2_norm = input_l2_norm
        self.output_l2_norm = output_l2_norm

        self._is_fitted = False

    def forward(self, features: list[torch.Tensor], **kwargs) -> AdapterOutput:
        assert len(features) == self.__num_inputs, f"Must passed same number of input as is initialized number of input projections, {len(features)} != {self.__num_inputs}"

        # Input projections into common hidden dim
        projected_inputs = []
        for in_features, input_projection in zip(features, self.input_projections):
            if self.input_l2_norm:
                in_features = F.normalize(in_features, p=2, dim=-1)

            h_input = input_projection(in_features)
            projected_inputs.append(h_input)

        # Stack projected inputs
        H = torch.stack(projected_inputs, dim=1) # (B, N inputs, D)
        # Modality dropout
        if self.training and self.__modality_dropout > 0.0:
            mask = torch.rand(H.shape[:2], device=H.device) > self.__modality_dropout
            H = H * mask.unsqueeze(-1)

        if self.__use_attn:
            H = H + self.input_pos_embed # Add positional embeddings of inputs
            Q = self.fusion_query.expand(H.size(0), -1, -1) # Add CLS token that will act as cross-attention pool
            Q_norm = self.q_norm(Q)
            H_norm = self.kv_norm(H)
            attn_out, _ = self.attn_fusion(Q_norm, H_norm, H_norm)
            Q = Q + self.post_attn_dropout(attn_out)
            Q = Q + self.post_ffn_dropout(self.attn_ffn(self.attn_ffn_norm(Q)))
            h_fused = Q.squeeze(1)
        else:
            H_flat = H.reshape(H.size(0), -1) # (B, N * D)
            H_concat = self.concat_proj(H_flat)
            H_gate = self.concat_gate(H_flat)
            H_residual = H.sum(dim=1) / (self.__num_inputs ** 0.5)
            H_concat = H_concat * H_gate
            h_fused = H_concat + self.concat_drop_path(H_residual)

        # Final (output) projection into desired dim
        h_z = h_fused + self.final_drop_path(self.final_mlp(self.final_norm(h_fused)))
        z_fused = self.output_layer(h_z)

        if self.output_l2_norm:
            z_fused = F.normalize(z_fused, p=2, dim=-1)

        return AdapterOutput(adapter_output=z_fused)

    def save_as_pretrained(self, save_path: str, allow_overwrite: bool = True) -> None:
        if not allow_overwrite:
            save_path = handle_existing_path(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "adapter_type": self.adapter_type,
                "config": {
                    "input_dims": self.__input_dims,
                    "hidden_dim": self.__hidden_dim,
                    "fused_dim": self.__fused_dim,
                    "dropout": self.__dropout,
                    "hidden_act": self.__hidden_act,
                    "use_attn": self.__use_attn,
                    "attn_num_heads": self.__attn_num_heads,
                    "attn_dropout": self.__attn_dropout,
                    "modality_dropout": self.__modality_dropout,
                    "drop_path_rate": self.__drop_path_rate,
                    "drop_path_mode": self.__drop_path_mode,
                    "input_norm": self.__input_norm,
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
            "attn_dropout": config.attn_dropout,
            "modality_dropout": config.modality_dropout,
            "drop_path_rate": config.drop_path_rate,
            "drop_path_mode": config.drop_path_mode,
            "input_norm": config.input_norm,
            "input_l2_norm": config.input_l2_norm,
            "output_l2_norm": config.output_l2_norm
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        return cls(**init_kwargs)