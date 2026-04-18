import os

from .config import FeatureAdapterConfig

from .pca import PCAAdapter
from .linear import LinearAdapter
from .fusion import FusionAdapter


ADAPTER_TYPE_DELIMITER = "---"


class AdapterWrapper:
    _adapters_mapping = {
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
    def from_pretrained(cls, adapter_model_name_or_path: str):
        adapter_type_and_path = adapter_model_name_or_path.split(ADAPTER_TYPE_DELIMITER, 1)
        assert len(adapter_type_and_path) == 2, f"Provide adapetr name in format `<adapter_type>---<adapter_path>` where adapter_type is one of {list(cls._adapters_mapping.keys())}"
        adapter_type, adapter_path = adapter_type_and_path
        adapter_cls = cls._adapters_mapping.get(adapter_type, None)
        assert adapter_cls is not None, f"Unsupported adapter type, must be one of {list(cls._adapters_mapping.keys())}"
        assert os.path.isfile(adapter_path), f"Provided path {adapter_path} does not lead to file"

        return adapter_cls.from_pretrained(adapter_path)

    @classmethod
    def from_config(cls, adapter_type: str, config: FeatureAdapterConfig):
        adapter_cls = cls._adapters_mapping.get(adapter_type, None)
        assert adapter_cls is not None, f"Unsupported adapter type, must be one of {list(cls._adapters_mapping.keys())}"

        return adapter_cls.from_config(config)