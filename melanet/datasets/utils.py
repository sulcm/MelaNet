from typing import Optional

from functools import partial
from collections import defaultdict

from PIL import ImageChops

from datasets import Dataset, DatasetDict
from datasets.features import Value, ClassLabel, Image
from datasets.combine import concatenate_datasets


def dataset_class_encode_column(dataset: Dataset, column: str, custom_labels: Optional[list[str]] = None, include_nulls: bool = False) -> Dataset:
    """Casts the given column as [`~datasets.features.ClassLabel`] and updates the table.

    Args:
        dataset (`Dataset`):
            `Dataset` object with `column` to be casted as `ClassLabel`
        column (`str`):
            The name of the column to cast (list all the column names with [`~datasets.Dataset.column_names`])
        custom_labels (`list[str] | None`):
            Custom label names to use. Defaults to `None` then unique values from `column` are used.
            All values from `column` must be found in `custom_labels` otherwise exception is raised.
        include_nulls (`bool`, defaults to `False`):
            Whether to include null values in the class labels. If `True`, the null values will be encoded as the `"None"` class label.

            <Added version="1.14.2"/>

    Example:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset("boolq", split="validation")
    >>> ds.features
    {'answer': Value('bool'),
        'passage': Value('string'),
        'question': Value('string')}
    >>> ds = ds.class_encode_column('answer')
    >>> ds.features
    {'answer': ClassLabel(num_classes=2, names=['False', 'True']),
        'passage': Value('string'),
        'question': Value('string')}
    ```
    """
    def __dataset_class_encode_column(dataset: Dataset, column: str, custom_labels: Optional[list[str]] = None, include_nulls: bool = False) -> Dataset:
        # Sanity checks
        if column not in dataset._data.column_names:
            raise ValueError(f"Column ({column}) not in table columns ({dataset._data.column_names}).")
        src_feat = dataset._info.features[column]
        if not isinstance(src_feat, (Value, ClassLabel)):
            raise ValueError(
                f"Class encoding is only supported for {Value.__name__} or {ClassLabel.__name__} column, and column {column} is {type(src_feat).__name__}."
            )

        if isinstance(src_feat, Value) and (src_feat.dtype != "string" or (include_nulls and None in dataset.unique(column))):

            def stringify_column(batch):
                batch[column] = [
                    str(sample) if include_nulls or sample is not None else None for sample in batch[column]
                ]
                return batch

            dset = dataset.map(
                stringify_column,
                batched=True,
                desc="Stringifying the column",
            )
        elif isinstance(src_feat, ClassLabel):

            def stringify_column(batch):
                batch[column] = [
                    src_feat.int2str(sample) if include_nulls or sample is not None else None for sample in batch[column]
                ]
                return batch

            old_features = dataset.features.copy()
            old_features[column] = Value("string")
            dset = dataset.map(
                stringify_column,
                batched=True,
                features=old_features,
                desc="Stringifying the `ClassLabel` column",
            )
        else:
            dset = dataset

        # Create the new feature
        if isinstance(custom_labels, list) and all([isinstance(l, str) for l in custom_labels]):
            _current_unique_labels = set([str(sample) for sample in dset.unique(column) if include_nulls or sample is not None])
            if not _current_unique_labels.issubset(custom_labels):
                raise ValueError("`custom_labels` does not include all labels for values found in `column`")

            class_names = custom_labels
            dst_feat = ClassLabel(names=class_names)
        else:
            class_names = sorted(str(sample) for sample in dset.unique(column) if include_nulls or sample is not None)
            dst_feat = ClassLabel(names=class_names)

        def cast_to_class_labels(batch):
            batch[column] = [
                dst_feat.str2int(str(sample)) if include_nulls or sample is not None else None
                for sample in batch[column]
            ]
            return batch

        new_features = dset.features.copy()
        new_features[column] = dst_feat

        dset = dset.map(
            cast_to_class_labels,
            batched=True,
            features=new_features,
            desc="Casting to class labels",
        )

        return dset

    if isinstance(dataset, DatasetDict):
        dataset._check_values_type()
        return DatasetDict(
            {k: __dataset_class_encode_column(dataset=ds, column=column, custom_labels=custom_labels, include_nulls=include_nulls) for k, ds in dataset.items()}
        )
    else:
        return __dataset_class_encode_column(dataset=dataset, column=column, custom_labels=custom_labels, include_nulls=include_nulls)


def validate_equal_datasets(left: Dataset, right: Dataset, columns: list[str], batch_size: int=256, num_proc: Optional[int]=2) -> tuple[bool, Optional[str]]:
    def __validate_equal_datasets(left: Dataset, right: Dataset, columns: list[str], batch_size: int=256, num_proc: Optional[int]=2) -> tuple[bool, Optional[str]]:
        columns_set = set(columns)
        if len(columns_set) != len(columns):
            return False, "In `columns` are repeating column names"
        if not (columns_set.issubset(left._data.column_names) and columns_set.issubset(right._data.column_names)):
            return False, "Not all specified columns are in datasets"
        if len(left) != len(right):
            return False, "Datasets are different lengths"
        if not all([type(left._info.features[column]) == type(right._info.features[column]) for column in columns]):
            return False, "Datasets columns have different feature types"

        column_comparator = {}
        # ==============================================================================================================
        # Equal functions
        def __eq_values(v_left, v_right) -> bool:
            return v_left == v_right

        def __eq_images(im_left, im_right) -> bool:
            diff = ImageChops.difference(im_left, im_right)
            return True if not diff.getbbox() else False

        def __eq_classlabel(l_left, l_right, l_cl: ClassLabel, r_cl: ClassLabel) -> bool:
            return l_cl.int2str(l_left) == r_cl.int2str(l_right)
        # ==============================================================================================================
        for col in columns:
            if isinstance(left._info.features[col], Value):
                column_comparator[col] = __eq_values
            elif isinstance(left._info.features[col], Image):
                column_comparator[col] = __eq_images
            elif isinstance(left._info.features[col], ClassLabel):
                column_comparator[col] = partial(
                    __eq_classlabel,
                    l_cl=left._info.features[col],
                    r_cl=right._info.features[col]
                )
            else:
                column_comparator[col] = __eq_values

        left2concat = left.select_columns(columns).rename_columns({c: f"left_{c}" for c in columns})
        right2concat = right.select_columns(columns).rename_columns({c: f"right_{c}" for c in columns})
        dset_concat = concatenate_datasets([left2concat, right2concat], axis=1)
        dset_concat: Dataset = dset_concat.add_column("__eq__", [False]*len(dset_concat))

        def iter_rows(batch: dict):
            keys = tuple(batch.keys())
            for values in zip(*batch.values()):
                yield dict(zip(keys, values))

        def map_compare(batch):
            batch["__eq__"] = [
                all([
                    column_comparator[c](sample[f"left_{c}"], sample[f"right_{c}"])
                    for c in columns
                ])
                for sample in iter_rows(batch)
            ]
            return batch

        dset_compared = dset_concat.map(
            map_compare,
            batched=True,
            desc="Comparing datasets row-wise",
            num_proc=num_proc,
            batch_size=batch_size
        )
        return (True, None) if all(dset_compared["__eq__"]) else (False, "There are differences between values in datasets")

    if isinstance(left, Dataset) and isinstance(right, Dataset):
        return __validate_equal_datasets(
            left=left,
            right=right,
            columns=columns,
            batch_size=batch_size,
            num_proc=num_proc
        )
    elif isinstance(left, DatasetDict) and isinstance(right, DatasetDict):
        split_results = [
            __validate_equal_datasets(
                left=l,
                right=r,
                columns=columns,
                batch_size=batch_size,
                num_proc=num_proc
            )
            for l, r in zip(left.values(), right.values())
        ]
        for res, m in split_results:
            if not res:
                return res, m
        return True, None
    else:
        return False, f"Datasets are not the same type: {type(left)} and {type(right)}"


def extend_datasets(datasets: list[DatasetDict]) -> DatasetDict:
    split2idx = defaultdict(list)
    for idx, d in enumerate(datasets):
        for split in d.keys():
            split2idx[split].append(idx)

    return DatasetDict({
        split: concatenate_datasets([datasets[idx][split] for idx in d_idx], split=split)
        for split, d_idx in split2idx.items()
    })