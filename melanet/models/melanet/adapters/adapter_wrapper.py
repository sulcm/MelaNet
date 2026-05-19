import os

from typing import Optional

from .base import BaseAdapter
from .learnable_base import LearnableAdapter
from .config import FeatureAdapterConfig

from .pca import PCAAdapter
from .linear import LinearAdapter
from .fusion import FusionAdapter


ADAPTER_TYPE_DELIMITER = "---"


class AdapterWrapper:
    _adapters_mapping: dict[str, type[BaseAdapter]] = {
        cls.adapter_type: cls
        for cls in (
            PCAAdapter,
            LinearAdapter,
            FusionAdapter,
        )
    }

    def __init__(self, *args, **kwargs):
        raise OSError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_pretrained(cls, adapter_model_name_or_path: str, adapter_type: Optional[str] = None):
        if ADAPTER_TYPE_DELIMITER in adapter_model_name_or_path:
            adapter_type_and_path = adapter_model_name_or_path.split(ADAPTER_TYPE_DELIMITER, 1)
            assert len(adapter_type_and_path) == 2, f"Provide adapetr name in format `<adapter_type>{ADAPTER_TYPE_DELIMITER}<adapter_path>` where adapter_type is one of {list(cls._adapters_mapping.keys())}"
            adapter_type, adapter_path = adapter_type_and_path
        elif adapter_type is not None:
            adapter_path = adapter_model_name_or_path
        else:
            raise ValueError(
                f"You must provide `adapter_model_name_or_path` and `adapter_type` OR adapetr name in format `<adapter_type>{ADAPTER_TYPE_DELIMITER}<adapter_path>` "
                f"where adapter_type is one of {list(cls._adapters_mapping.keys())}"
            )

        adapter_cls = cls._adapters_mapping.get(adapter_type, None)
        assert adapter_cls is not None, f"Unsupported adapter type, must be one of {list(cls._adapters_mapping.keys())}"
        assert os.path.isfile(adapter_path), f"Provided path {adapter_path} does not lead to file"

        return adapter_cls.from_pretrained(pretrained_path=adapter_path)

    @classmethod
    def from_config(cls, config: FeatureAdapterConfig, adapter_type: str):
        adapter_cls = cls._adapters_mapping.get(adapter_type, None)
        assert adapter_cls is not None, f"Unsupported adapter type, must be one of {list(cls._adapters_mapping.keys())}"

        return adapter_cls.from_config(config=config)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: FeatureAdapterConfig, adapter_type: str):
        adapter_cls = cls._adapters_mapping.get(adapter_type, None)
        assert adapter_cls is not None, f"Unsupported adapter type, must be one of {list(cls._adapters_mapping.keys())}"
        assert issubclass(adapter_cls, LearnableAdapter), "Loading from checkpoint is supported only for `LearnableAdapter` types"

        return adapter_cls.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config
        )