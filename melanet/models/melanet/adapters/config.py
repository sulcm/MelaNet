from typing import Optional, Union
from pydantic import BaseModel, Field


class FeatureAdapterConfig(BaseModel):
    out_features: int = Field(...)
    in_features: Union[int, list[int], None] = Field(default=None)
    bias: Optional[bool] = Field(default=None)
    input_l2_norm: Optional[bool] = Field(default=None)
    output_l2_norm: Optional[bool] = Field(default=None)
    input_norm: Optional[bool] = Field(default=None)
    whiten: Optional[bool] = Field(default=None)
    hidden_dim: Optional[int] = Field(default=None)
    dropout: Optional[float] = Field(default=None)
    modality_dropout: Optional[float] = Field(default=None)
    hidden_act: Optional[str] = Field(default=None)
    use_attn: Optional[bool] = Field(default=None)
    attn_num_heads: Optional[int] = Field(default=None)
    attn_dropout: Optional[float] = Field(default=None)
    drop_path_rate: Optional[float] = Field(default=None)
    drop_path_mode: Optional[str] = Field(default=None)
    seed: Optional[int] = Field(default=42)

    @classmethod
    def from_cli(cls, cli_options: str) -> "FeatureAdapterConfig":
        init_kwargs = {}
        if cli_options is not None:
            for mapping in cli_options.replace(" ", "").split(","):
                key, value = mapping.split("=")
                init_kwargs[key] = value
        return cls(**init_kwargs)


def create_default_feature_adapter_config(out_features: int, **kwargs) -> FeatureAdapterConfig:
    def_feature_adapter_cfg = FeatureAdapterConfig(
        out_features=out_features,
        **kwargs
    )

    return def_feature_adapter_cfg