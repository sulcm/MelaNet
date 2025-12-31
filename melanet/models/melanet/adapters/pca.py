import numpy as np
import pickle

from typing import Optional

from sklearn.decomposition import PCA

from .base import BaseAdapter
from ..embeddings.utils import normalize_embeddings


class PCAAdapter(BaseAdapter):
    def __init__(
        self,
        out_features: int,
        whiten: bool = False,
        seed: Optional[int] = 42,
        normalize: bool = False
    ):
        self.pca = PCA(
            n_components=out_features,
            whiten=whiten,
            random_state=seed
        )
        self.normalize = normalize
        self._is_fitted = False

    def forward(self, features: np.ndarray) -> np.ndarray:
        assert self._is_fitted, "PCAAdapter is not fitted. Call `fit()` first."

        proj_features = self.pca.transform(features)
        if self.normalize:
            proj_features = normalize_embeddings(proj_features)
        return proj_features

    def __call__(self, features: np.ndarray, **kwargs) -> np.ndarray:
        return self.forward(features)

    def fit(self, X: np.ndarray, y = None) -> None:
        self.pca.fit(X)
        self._is_fitted = True

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "PCAAdapter":
        with open(pretrained_path, "rb") as f:
            obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj

    def save_as_pretrained(self, save_path: str) -> None:
        with open(save_path, "wb") as f:
            pickle.dump(self, f)