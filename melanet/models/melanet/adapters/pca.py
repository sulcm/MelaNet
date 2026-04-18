import os
import pickle
import numpy as np

from typing import Optional

from sklearn.decomposition import PCA

from .base import BaseAdapter, DummyNonLearnableAdapter
from .types import AdapterOutput
from .config import FeatureAdapterConfig
from ..embeddings.utils import normalize_embeddings
from ..utils import handle_existing_path


class PCAAdapter(BaseAdapter, DummyNonLearnableAdapter):
    adapter_type = "pca"

    def __init__(
        self,
        out_features: int,
        whiten: bool = False,
        seed: Optional[int] = 42,
        input_l2_norm: bool = False,
        output_l2_norm: bool = False
    ):
        self.pca = PCA(
            n_components=out_features,
            whiten=whiten,
            random_state=seed
        )

        self.input_l2_norm = input_l2_norm
        self.output_l2_norm = output_l2_norm

        self._is_fitted = False

    def forward(self, features: np.ndarray, **kwargs) -> AdapterOutput:
        if self.input_l2_norm:
            features = normalize_embeddings(features)

        proj_features = self.pca.transform(features)

        if self.output_l2_norm:
            proj_features = normalize_embeddings(proj_features)
        return AdapterOutput(adapter_output=proj_features)

    def __call__(self, features: np.ndarray, **kwargs) -> AdapterOutput:
        return self.forward(features)

    def fit(self, X: np.ndarray, y = None) -> "PCAAdapter":
        if self.input_l2_norm:
            X = normalize_embeddings(X)

        self.pca.fit(X)
        self._is_fitted = True

        return self

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "PCAAdapter":
        with open(pretrained_path, "rb") as f:
            obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj

    def save_as_pretrained(self, save_path: str, allow_overwrite: bool = True) -> None:
        if not allow_overwrite:
            save_path = handle_existing_path(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_config(cls, config: FeatureAdapterConfig) -> "PCAAdapter":
        init_kwargs = {
            "out_features": config.out_features,
            "whiten": config.whiten,
            "input_l2_norm": config.input_l2_norm,
            "output_l2_norm": config.output_l2_norm,
            "seed": config.seed
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None or k == "seed"}
        return cls(**init_kwargs)