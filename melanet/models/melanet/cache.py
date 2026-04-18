import os
import pickle
import traceback
import numpy as np
import pandas as pd

from typing import TypeVar, Generic, Optional, Any, TypedDict
from dataclasses import dataclass


T = TypeVar("T")

class ClassifierCache(TypedDict):
    logits: dict[str, list[np.ndarray]]

class FeatureExtractorCache(TypedDict):
    index: pd.DataFrame
    eval_dataset: pd.DataFrame

MetadataType = dict[str, Any]


@dataclass
class CacheManager(Generic[T]):
    """
    Cache container.

    Internal structure:
    {
        "cache": T,
        "metadata": MetadataType
    }
    """

    _cache: T
    _metadata: MetadataType

    @property
    def cache(self) -> T:
        return self._cache

    @property
    def metadata(self) -> MetadataType:
        return self._metadata

    @staticmethod
    def save(
        *,
        path: str,
        cache: T,
        metadata: Optional[MetadataType] = None
    ) -> None:
        """
        Save arbitrary cache object and optional metadata.
        """
        try:
            if metadata is not None and not isinstance(metadata, dict):
                raise TypeError("Metadata must be a dictionary or `None`.")

            payload = {
                "cache": cache,
                "metadata": {
                    **(metadata or {})
                }
            }
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            print(f"ERROR - CacheManager - {traceback.format_exc()}")

    @classmethod
    def load(
        cls,
        path: str
    ) -> "CacheManager[T]":
        """
        Load and validate cache file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "rb") as f:
            payload = pickle.load(f)

        if "cache" not in payload:
            raise ValueError("Invalid cache structure.")

        if "metadata" not in payload:
            payload["metadata"] = {}
        elif not isinstance(payload["metadata"], dict):
            raise ValueError("metadata must be a `dict`.")

        return cls(
            _cache=payload["cache"],
            _metadata=payload["metadata"]
        )