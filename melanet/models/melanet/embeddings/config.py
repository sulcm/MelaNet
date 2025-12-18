from typing import Optional, Literal
from pydantic import BaseModel, Field


class FeatureExtractorConfig(BaseModel):
    output_type: Literal["object", "sum", "concat"] = Field(
        default="object",
        description="""Specify output format:
        - "object": returns `ZeroShotOutput` object
        - "sum": applies weighted sum of embeddings
        - "concat": concatenates embeddings into single vector `[ft_embeds, zero_shot_embeds]`"""
    )
    alpha: Optional[float] = Field(
        default=None,
        description="""Used when `output_type="sum"` in form $\alpha * ft_embeds + (1 - \alpha) * zero_shot_embeds$"""
    )

    @classmethod
    def from_cli(cls, cli_options: str) -> "FeatureExtractorConfig":
        init_kwargs = {}
        for mapping in cli_options.replace(" ", "").split(","):
            key, value = mapping.split("=")
            init_kwargs[key] = value
        return cls(**init_kwargs)


def create_default_feature_extractor_config() -> FeatureExtractorConfig:
    def_feature_extractor_cfg = FeatureExtractorConfig()

    if def_feature_extractor_cfg.output_type == "sum":
        assert def_feature_extractor_cfg.alpha is not None

    return def_feature_extractor_cfg