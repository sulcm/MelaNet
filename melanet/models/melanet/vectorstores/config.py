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
    top_k: Optional[int] = Field(
        default=1,
        description="From retrieved results select first top K results."
    )
    search_k: Optional[int] = Field(
        default=20,
        description="Search for K closest elements in index. Only applied if `top_k > 1`."
    )
    return_scores: Optional[bool] = Field(
        default=False,
        description="Whether to return tuple `[(class, score), ...]` or only `[class, ...]`."
    )
    rerank_top_n: Optional[int] = Field(
        default=1,
        description="Return top N reranked results."
    )
    rrf_k: Optional[float] = Field(
        default=60.0,
        description="Parameter K for RRF computation."
    )

    @classmethod
    def from_cli(cls, cli_options: str) -> "VectorStoreConfig":
        init_kwargs = {}
        if cli_options is not None:
            for mapping in cli_options.replace(" ", "").split(","):
                key, value = mapping.split("=")
                init_kwargs[key] = value
        return cls(**init_kwargs)


def create_default_vector_store_config(**kwargs) -> VectorStoreConfig:
    def_vector_store_cfg = VectorStoreConfig(**kwargs)

    return def_vector_store_cfg