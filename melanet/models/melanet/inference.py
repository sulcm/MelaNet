import torch
import numpy as np
import pandas as pd

from typing import Any, Callable, Union, Literal
from tqdm import tqdm
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import Dataset

from .melanet_wrapper import MelaNet
from .embeddings.types import FeatureExtractorOutput, MELANET_FEATURES_PREFIX
from .zero_shot.augmentations import create_views_batched
from .utils import tensor2numpy


class InferenceDataCollator():
    def __init__(
        self,
        image_column_name: str,
        id_column_name: str,
        transforms: list[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        self.image_column_name = image_column_name
        self.id_column_name = id_column_name

        self.transforms = transforms
        self.num_views = (len(transforms) + 1) if transforms else 1

    def __call__(self, batch):
        images = [sample[self.image_column_name] for sample in batch]
        lesion_ids = np.array([sample[self.id_column_name] for sample in batch])

        if self.transforms:
            images = create_views_batched(images, self.transforms)
            lesion_ids = lesion_ids.repeat(self.num_views)

        return {
            self.image_column_name: images,
            self.id_column_name: lesion_ids
        }


def run_inference(
    model: MelaNet,
    dataset: Dataset,
    image_column_name: str,
    id_column_name: str,
    model_kwargs: dict[str, Any] = None,
    batch_size: int = 1,
    transforms: list[Callable[[torch.Tensor], torch.Tensor]] = None,
    num_workers: int = 1,
    pin_memory: bool = True,
    device_type: str = "cuda",
) -> dict[str, Union[list[np.ndarray], list[pd.DataFrame]]]:
    if transforms:
        real_batch_size = batch_size // (len(transforms) + 1)
    else:
        real_batch_size = batch_size
    assert real_batch_size > 0

    model_kwargs = (model_kwargs or {})

    collator = InferenceDataCollator(
        image_column_name=image_column_name,
        id_column_name=id_column_name,
        transforms=transforms
    )
    dataloader = DataLoader(
        dataset,
        batch_size=real_batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=pin_memory,
        collate_fn=collator
    )

    predictions = {} if model.is_feature_extractor and model.feature_extractor_config.output_type == "object" else defaultdict(list)
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Batch"):
            images = batch[image_column_name]
            lesion_ids = batch[id_column_name]

            with torch.autocast(device_type=device_type):
                outputs = model(image=images, **model_kwargs)

            if model.is_feature_extractor:
                if model.feature_extractor_config.output_type == "object":
                    _pd_out = outputs.to_pandas()
                    _pd_out[id_column_name] = lesion_ids
                    _pd_out[id_column_name] = _pd_out[id_column_name].astype(str)
                    _pd_out_grouped = _pd_out.groupby(id_column_name)
                    for group in _pd_out_grouped.groups:
                        if group in predictions:
                            predictions[group] = pd.concat([predictions[group], _pd_out_grouped.get_group(group)], ignore_index=True)
                        else:
                            predictions[group] = _pd_out_grouped.get_group(group)
                else:
                    for lesion_id, output in zip(lesion_ids, outputs):
                        predictions[str(lesion_id)].append(
                            tensor2numpy(output)
                        )
            else:
                for lesion_id, output in zip(lesion_ids, outputs):
                    predictions[str(lesion_id)].append(output)
    return predictions


def output_to_dataframe(
    output: dict[str, Union[list[np.ndarray], list[pd.DataFrame]]],
    id_column_name: str,
    output_type: Literal["features", "object", "logits", "probs"],
) -> pd.DataFrame:
    if output_type == "object":
        pd_output = pd.concat(output.values(), ignore_index=True)

        _feature_names = FeatureExtractorOutput.get_feature_names()
        _to_drop = []
        _to_rename = []
        for f_name in _feature_names:
            if pd_output[f_name].hasnans:
                _to_drop.append(f_name)
            else:
                _to_rename.append(f_name)

        if _to_drop:
            pd_output = pd_output.drop(columns=_to_drop)
        if _to_rename:
            pd_output = pd_output.rename(
                columns={f_name: f"{MELANET_FEATURES_PREFIX}_{f_name}" for f_name in _to_rename}
            )
    elif output_type in ("features", "logits", "probs"):
        values_name = MELANET_FEATURES_PREFIX if output_type == "features" else output_type
        pd_output = pd.Series(
            output
        ).explode().reset_index(name=values_name).rename(columns={"index": id_column_name})
    else:
        raise ValueError(f"Unsupported output type {output_type}")
    return pd_output