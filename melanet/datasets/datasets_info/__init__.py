from .base import BaseDatasetInfo
from .milk10k import MILK10kInfo
from .isic2019 import ISIC2019Info
from .melano_mix import MelanoMixInfo


DATASET_INFO_MAPPING: dict[str, type[BaseDatasetInfo]] = {
    "milk10": MILK10kInfo,
    "isic2019": ISIC2019Info,
    "melano_mix": MelanoMixInfo
}

def get_dataset_info(dataset_name: str) -> BaseDatasetInfo:
    assert dataset_name in DATASET_INFO_MAPPING, f"Provided dataset name {dataset_name} is not in known datasets. Must be one of {list(DATASET_INFO_MAPPING.keys())}"
    return DATASET_INFO_MAPPING[dataset_name]()