from pydantic import BaseModel, Field

from typing import Literal, Union, Optional


SUPPORTED_METRICS = Literal["ip", "l2"]

class VectorStoreConfig(BaseModel):
    metric: Union[SUPPORTED_METRICS, list[SUPPORTED_METRICS]] = Field(
        default="ip",
        description="Metric for similarity search in created index. Defaults to 'ip'."
    )
    pca_components: Union[Optional[int], list[Optional[int]]] = Field(
        default=None,
        description="Optional use of PCA for dimension reduction during index build and search. Turn OFF using `None`."
    )

    @classmethod
    def from_cli(cls, cli_options: str) -> "VectorStoreConfig":
        init_kwargs = {}
        for mapping in cli_options.replace(" ", "").split(","):
            key, value = mapping.split("=")
            init_kwargs[key] = value
        return cls(**init_kwargs)


def create_default_vector_store_config() -> VectorStoreConfig:
    def_vector_store_cfg = VectorStoreConfig()

    return def_vector_store_cfg