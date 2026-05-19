from typing import Literal
from abc import ABC, abstractmethod

from .config import FeatureAdapterConfig
from .types import AdapterOutput


ADAPTER_TYPES = ("pca", "linear", "fusion")


class BaseAdapter(ABC):
    adapter_type: Literal["pca", "linear", "fusion"]
    _is_fitted: bool = False

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseAdapter":
        ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> AdapterOutput:
        ...

    @classmethod
    @abstractmethod
    def from_pretrained(cls, pretrained_path: str) -> "BaseAdapter":
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: FeatureAdapterConfig) -> "BaseAdapter":
        ...

    @abstractmethod
    def save_as_pretrained(self, save_path: str, allow_overwrite: bool = True) -> None:
        ...


class DummyNonLearnableAdapter(ABC):
    """Dummy implementations of some `torch.nn.Module` methods"""

    def to(self, *args, **kwargs):
        return self

    def train(self, *args, **kwargs):
        return self

    def eval(self, *args, **kwargs):
        return self