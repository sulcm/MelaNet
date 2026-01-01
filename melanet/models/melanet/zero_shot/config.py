from typing import Optional, Literal
from pydantic import BaseModel, Field


class ZeroShotConfig(BaseModel):
    model_backend: Literal["hf", "open_clip", "timm"]
    output_type: Literal["sum", "concat"] = Field(
        default="concat",
        description="""Specify output format:
        - "sum": applies weighted sum of embeddings
        - "concat": concatenates embeddings into single vector `[image_embeds, text_embeds]`"""
    )
    normalize_output: Optional[bool] = Field(
        default=True,
        description="Whether to apply L2 normalization to embeddings vectors. Defaults to `True`."
    )
    alpha: Optional[float] = Field(
        default=None,
        description="""Used when `output_type="sum"` in form $\alpha * image_embeds + (1 - \alpha) * text_embeds$"""
    )
    add_text_embeddings: Optional[bool] = Field(
        default=False,
        description="Process text inputs. Useful for models with image-text alignment such as CLIP."
    )
    global_pool: Optional[str] = Field(
        default="avg",
        description="Used in inicialization of `timm` models."
    )
    features_output_name: Optional[str] = Field(
        default="pooler_output",
        description="Used for extracting features from model output object. If `None` assume the features are the model output."
    )

    @classmethod
    def from_cli(cls, cli_options: str) -> "ZeroShotConfig":
        init_kwargs = {}
        if cli_options is not None:
            for mapping in cli_options.replace(" ", "").split(","):
                key, value = mapping.split("=")
                init_kwargs[key] = value
        return cls(**init_kwargs)


def create_default_zero_shot_config(model_backend: str, **kwargs) -> ZeroShotConfig:
    def_zero_shot_cfg = ZeroShotConfig(
        model_backend=model_backend,
        **kwargs
    )

    if def_zero_shot_cfg.output_type == "sum":
        assert def_zero_shot_cfg.alpha is not None

    return def_zero_shot_cfg