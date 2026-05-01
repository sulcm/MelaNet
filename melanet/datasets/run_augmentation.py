import os
import logging
import numpy as np

from typing import cast, Optional
from dataclasses import dataclass, field
from collections import Counter

from tqdm import tqdm

from transformers import set_seed, HfArgumentParser
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from datasets.combine import concatenate_datasets

from torchvision.transforms import v2 as torch_augmentation

from metacentrum_utils import METACENTRUM_SCRATCH_PREFIX, load_dataset_from_scratch


logger = logging.getLogger(__name__)


@dataclass
class DataAugmentationArguments:
    """
    Arguments pertaining to how augment data in dataset
    """

    dataset_name: str = field(
        metadata={"help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."},
    )
    augmented_dataset_path: str = field(
        metadata={"help": "Path where to save augmented dataset."}
    )
    augment_splits: Optional[str] = field(
        default=None,
        metadata={
            "help": "If `None` then assume its single dataset of type `Dataset` otherwise provide name of the split or splits (e.g. 'train', 'train+validation', etc.)."
        }
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "The batch size used for application of augmentations."}
    )
    num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes to run augmentations on."}
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."},
    )
    label_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed that will be set to ensure reproducibility. Defaults to 42."}
    )


def _get_basic_augmentation_transforms_pipeline(
    dataset: Dataset,
    image_column_name: str = "image",
    label_column_name: str = "label"
):
    image_resolutions = set([pil_im.size for pil_im in dataset[image_column_name]])

    return torch_augmentation.Compose([
        torch_augmentation.RandomHorizontalFlip(),
        torch_augmentation.RandomRotation([-360, 360]),
        torch_augmentation.ColorJitter(brightness=0.1, contrast=0.1),
        torch_augmentation.RandomChoice([
            torch_augmentation.RandomResizedCrop(size, scale=(0.8, 1.0)) for size in image_resolutions
        ])
    ])


def _augment_split(augment_args: DataAugmentationArguments, dataset: Dataset, split_name: Optional[str] = None) -> Dataset:
    labels_counts = Counter(dataset[augment_args.label_column_name])
    most_common_label_id, max_label_count = labels_counts.most_common(1)[0]

    offline_augmentations = _get_basic_augmentation_transforms_pipeline(
        dataset=dataset,
        image_column_name=augment_args.image_column_name,
        label_column_name=augment_args.label_column_name
    )

    def map_augmentations(batch):
        # Workaround for torchvision.transforms applying same RNG state for whole batch -> iterate thru batch
        batch[augment_args.image_column_name] = [
            offline_augmentations(im.convert("RGB"))
            for im in batch[augment_args.image_column_name]
        ]
        return batch

    ds_per_label: list[Dataset] = []
    for l, c in tqdm(labels_counts.items(), unit="label", desc="Dataset augmentation"):
        _cls_samples = dataset.filter(lambda cols: [v == l for v in cols[augment_args.label_column_name]], batched=True)
        if l != most_common_label_id:
            repeat_count = (max_label_count // c) - 1
            if repeat_count > 0:
                # Repeat class in full lenght
                repeated_samples = _cls_samples.repeat(repeat_count)
            else:
                # Create empty placeholder
                repeated_samples = _cls_samples.repeat(0)

            if max_label_count - (c + len(repeated_samples)) >= max_label_count // 100:
                # Fill-out remaining samples if there is greater then a hundredth difference in lenght between most common class and current one
                _rnd_select = np.random.choice(
                    len(_cls_samples),
                    ((max_label_count - (c + len(repeated_samples))) // 8) * 8,
                    replace=False
                )
                padding_samples = _cls_samples.select(_rnd_select)
                repeated_samples = concatenate_datasets([repeated_samples, padding_samples])

            repeated_samples = repeated_samples.map(
                map_augmentations,
                batched=True,
                batch_size=augment_args.batch_size,
                num_proc=augment_args.num_proc,
                desc=f"Augmentation of repeated samples from {l} class"
            )
            logger.info(f"Adding class {l} (with augmentations)")
            ds_per_label.append(
                concatenate_datasets([_cls_samples, repeated_samples])
            )
        else:
            logger.info(f"Adding most common class {l} (without augmentations)")
            ds_per_label.append(_cls_samples)

    augmented_dataset = concatenate_datasets(ds_per_label)
    augmented_dataset = augmented_dataset.shuffle(seed=augment_args.seed)
    return augmented_dataset


def augment_dataset(augment_args: DataAugmentationArguments):
    assert os.path.isdir(os.path.dirname(augment_args.augmented_dataset_path)), "Parent directory must exist."
    if os.path.isdir(augment_args.augmented_dataset_path):
        logger.warning(f"Path {augment_args.augmented_dataset_path} already exists and will be overwritten.")

    if os.path.exists(augment_args.dataset_name):
        # Load from local path
        logger.info(f"Loading dataset {augment_args.dataset_name} from local path")
        dataset = load_from_disk(
            dataset_path=augment_args.dataset_name
        )
    elif augment_args.dataset_name.startswith(METACENTRUM_SCRATCH_PREFIX):
        # Load from scratch directory on Metacentrum
        logger.info(f"Loading dataset {augment_args.dataset_name} from scratch storage")
        dataset = load_dataset_from_scratch(
            augment_args.dataset_name
        )
    else:
        # Pull from HF or load it from cache
        logger.info(f"Loading dataset {augment_args.dataset_name} from HF Hub")
        dataset = load_dataset(
            augment_args.dataset_name
        )
    _ = dataset.cleanup_cache_files()

    if not isinstance(dataset, Dataset):
        assert augment_args.augment_splits is not None, "When using dataset with multiple splits provide split name(s)"
        augment_splits = augment_args.augment_splits.split("+")
    else:
        augment_splits = []

    # Set seed (before running augmentations)
    if augment_args.seed is not None:
        set_seed(augment_args.seed)

    if isinstance(dataset, Dataset):
        if augment_args.image_column_name not in dataset.column_names:
            raise ValueError(
                f"--image_column_name {augment_args.image_column_name} not found in dataset '{augment_args.dataset_name}'. "
                "Make sure to set `--image_column_name` to the correct image column - one of "
                f"{', '.join(dataset.column_names)}."
            )
        if augment_args.label_column_name not in dataset.column_names:
            raise ValueError(
                f"--label_column_name {augment_args.label_column_name} not found in dataset '{augment_args.dataset_name}'. "
                "Make sure to set `--label_column_name` to the correct text column - one of "
                f"{', '.join(dataset.column_names)} - or remove it and skip metrics computations (no ground truth)."
            )

        augmented_dataset = _augment_split(
            augment_args=augment_args,
            dataset=dataset
        )
    else:
        for dset_split in augment_splits:
            assert dset_split in dataset, f"Split {dset_split} was not found in dataset, must be some of {dataset.keys()}"
            if augment_args.image_column_name not in dataset[dset_split].column_names:
                raise ValueError(
                    f"--image_column_name {augment_args.image_column_name} not found in dataset '{augment_args.dataset_name}'. "
                    "Make sure to set `--image_column_name` to the correct image column - one of "
                    f"{', '.join(dataset[dset_split].column_names)}."
                )
            if augment_args.label_column_name not in dataset[dset_split].column_names:
                raise ValueError(
                    f"--label_column_name {augment_args.label_column_name} not found in dataset '{augment_args.dataset_name}'. "
                    "Make sure to set `--label_column_name` to the correct text column - one of "
                    f"{', '.join(dataset[dset_split].column_names)} - or remove it and skip metrics computations (no ground truth)."
                )

        augmented_dataset = DatasetDict({
            split: _augment_split(
                augment_args=augment_args,
                dataset=dataset[split],
                split_name=split
            )
            if split in augment_splits else
            dataset[split]
            for split in dataset.keys()
        })

    logger.info(f"Saving augmented dataset to {augment_args.augmented_dataset_path}")
    augmented_dataset.save_to_disk(augment_args.augmented_dataset_path)


if __name__ == "__main__":
    parser = HfArgumentParser(DataAugmentationArguments)
    augment_args = cast(
        DataAugmentationArguments,
        parser.parse_args_into_dataclasses()[0]
    )

    augment_dataset(augment_args=augment_args)