from typing import Optional
from pydantic import BaseModel, Field


class FeatureAdapterConfig(BaseModel):
    out_features: int = Field(...)
    in_features_A: Optional[int] = Field(default=None)
    in_features_B: Optional[int] = Field(default=None)
    bias: Optional[bool] = Field(default=False)
    input_l2_norm: Optional[bool] = Field(default=False)
    output_l2_norm: Optional[bool] = Field(default=True)
    whiten: Optional[bool] = Field(default=False)
    hidden_dim: Optional[int] = Field(default=1024)
    dropout: Optional[float] = Field(default=0.1)
    hidden_act: Optional[str] = Field(default="gelu")
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