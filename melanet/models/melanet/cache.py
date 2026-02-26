import os
import torch
import traceback
import numpy as np

from typing import TypeVar, Generic, Optional, Union, Any, TypedDict
from dataclasses import dataclass


T = TypeVar("T")

class ClassifierCache(TypedDict):
    logits: np.ndarray

class FeatureExtractorCache(TypedDict):
    index: dict[str, torch.Tensor]
    eval_dataset: dict[str, torch.Tensor]


@dataclass
class CacheManager(Generic[T]):
    """
    Cache container.

    Internal structure:
    {
        "cache": T,
        "metadata": dict[str, Any]
    }
    """

    _cache: T
    _metadata: dict[str, Any]

    @property
    def cache(self) -> T:
        return self._cache

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @staticmethod
    def save(
        *,
        path: str,
        cache: T,
        metadata: Optional[dict[str, Any]] = None
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
            torch.save(payload, path)
        except Exception:
            print(f"ERROR - CacheManager - {traceback.format_exc()}")

    @classmethod
    def load(
        cls,
        path: str,
        device: Union[str, torch.device] = "cpu"
    ) -> "CacheManager[T]":
        """
        Load and validate cache file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        payload = torch.load(path, map_location=device, weights_only=False)

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