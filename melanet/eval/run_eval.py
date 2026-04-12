import os
import sys
import json
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from typing import cast, Optional, Callable, Literal
from dataclasses import dataclass, field
from datetime import datetime

from transformers import HfArgumentParser
from datasets import load_from_disk, load_dataset, Dataset
from datasets.combine import concatenate_datasets

from torchmetrics import (
    AUROC,
    AveragePrecision,
    Accuracy,
    Specificity,
    Recall,
    F1Score,
    Precision,
    NegativePredictiveValue
)

from melanet.melanet_wrapper import MelaNet, FeatureExtractorConfig, ZeroShotConfig, FeatureExtractorOutput
from melanet.cache import CacheManager, ClassifierCache, FeatureExtractorCache
from melanet.embeddings.types import MELANET_FEATURES_PREFIX
from melanet.vectorstores import VectorStoreConfig, create_default_vector_store_config, MultiNNClassifier, NNClassifier
from melanet.vectorstores.rrf import reciprocal_rank_fusion
from melanet.zero_shot.augmentations import build_view_transformations
from melanet.functional.softmax import softmax
from melanet.utils import tensor2value, kwargs2cli
from melanet.inference import run_inference
from formatter.isic import format_isic_submission
from metacentrum_utils import DATASET_SCRATCH_PREFIX, load_dataset_from_scratch


logger = logging.getLogger(__name__)
log_levels = logging.getLevelNamesMapping()


@dataclass
class EvaluateArguments:
    """
    Arguments pertaining to how evaluate given model on dataset
    """

    classification_task: Literal["binary", "multiclass", "multilabel"] = field(
        metadata={"help": "Specify classification task / objective. Possible values are 'binary', 'multiclass', 'multilabel', or 'feature_classification'."}
    )
    eval_as_feature_extraction: bool = field(
        metadata={"help": "Init model and processing as classifier or feature extractor with vectorstores."}
    )
    dataset_name: str = field(
        metadata={"help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."},
    )
    results_path: str = field(
        metadata={"help": "Path and file where to save results. Must be JSON file."}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to finetuned model or model identifier from huggingface.co/models. Can be `None` for `feature_classification` when using zero-shot model."},
    )
    zero_shot_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-trained zero-shot model or model identifier from huggingface.co/models. Only used when `feature_classification` is active."},
    )
    prediction_resolution_strategy: Literal["greedy", "rrf", "first", "last", "mean"] = field(
        default="greedy",
        metadata={"help": "Strategy how to resolve predictions that have same ID or when using TTA transform strategies during evaluation."},
    )
    load_cached_model_inference: Optional[str] = field(
        default=None,
        metadata={"help": "Provide path from where to load cached infered values from model."},
    )
    cache_model_inference: Optional[str] = field(
        default=None,
        metadata={"help": "Provide path where to store / cache infered values from model."},
    )
    isic_submission_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path where to save formatted CSV submission for ISIC."},
    )
    eval_split: Optional[str] = field(
        default=None,
        metadata={
            "help": "If `None` then assume its single dataset of type `Dataset` otherwise provide name of the split or splits (e.g. 'test', 'validation+test', etc.)."
        }
    )
    index_name: str = field(
        default=None,
        metadata={"help": (
            "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)"
            " that will be used to create index for zero-shot classification."
            )
        },
    )
    index_split: Optional[str] = field(
        default=None,
        metadata={"help": "Used for `feature_classification` to build index."}
    )
    index_size: Optional[float] = field(
        default=None,
        metadata={"help": (
            "Specify size of build index."
            " - <1.0; inf) randomly selects N samples"
            " - (0.0; 1.0) randomly selects given portion of dataset"
            " - `None` then use full dataset"
            )
        }
    )
    feature_extractor_config: Optional[str] = field(
        default=None,
        metadata={"help": "Used for configuring feature extraction process."}
    )
    zero_shot_config: Optional[str] = field(
        default=None,
        metadata={"help": "Used for configuring zero-shot process."}
    )
    apply_augmentations: Optional[str] = field(
        default=None,
        metadata={"help": "Use augmentations during all phases. Overrides `index_augmentations` and `test_time_augmentations`. Defaults to `None` -> off. Can be 'all' or list of augmentations in format 'augment_1,augment_2,...'."}
    )
    index_augmentations: Optional[str] = field(
        default=None,
        metadata={"help": "Use augmentations during creation of index. Defaults to `None` -> off. Can be 'all' or list of augmentations in format 'augment_1,augment_2,...'."}
    )
    test_time_augmentations: Optional[str] = field(
        default=None,
        metadata={"help": "Use augmentations during test time. Defaults to `None` -> off. Can be 'all' or list of augmentations in format 'augment_1,augment_2,...'."}
    )
    feature_adapter: Optional[Literal["pca", "linear", "fusion"]] = field(
        default=None,
        metadata={"help": (
            "Create adapter that will be used for transforming extracted features. Possible options:"
            " - 'pca': Initialize PCA adapter that will select N components (lower dimension)"
            " - 'linear': Initialize simple linear projection layer that will simply transform embedding of one size to another"
            " - 'fusion': Initialize MLP that will combine embeddings from 2 models into single representation"
            )
        }
    )
    feature_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Provide path from or where feature adapter will be loaded and/or saved."}
    )
    feature_adapter_config: Optional[str] = field(
        default=None,
        metadata={"help": "Used for configuring adapters."}
    )
    vector_store_config: Optional[str] = field(
        default=None,
        metadata={"help": "Used for configuring vector stores."}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "The batch size used for inference"}
    )
    id_column_name: str = field(
        default="id",
        metadata={"help": "The name of the dataset column containing the IDs of samples. Defaults to 'id'."},
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the dataset column containing the labels. "
                "If `None` then return list of predicted labels. "
                "If column label is provided then among predicted labels compute and writes metrics."
            )
        },
    )
    default_log_level = "info"
    log_level: Literal[*log_levels.keys()] = field( # type:ignore
        default=default_log_level,
        metadata={"help": f"Set logging level. Must be one of {list(log_levels.keys())}"},
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Select which device to use for inference."}
    )
    num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of workers that will be used for loading data."}
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed that will be set to ensure reproducibility. Defaults to 42."}
    )

    def __override_none_args(self, arg_name: str):
        str_none_values = ("none", "null", "nan")
        num_none_values = (-1,)

        current_value = getattr(self, arg_name, None)
        if current_value is None:
            return
        elif isinstance(current_value, str):
            if current_value.lower() in str_none_values:
                setattr(self, arg_name, None)
        elif isinstance(current_value, (int, float)):
            if current_value in num_none_values:
                setattr(self, arg_name, None)
        else:
            pass

    def __post_init__(self):
        self.__override_none_args("apply_augmentations")
        self.__override_none_args("index_augmentations")
        self.__override_none_args("test_time_augmentations")


def _get_classification_metrics(
    task: Literal["binary", "multiclass", "multilabel"],
    num_classes: Optional[int] = None,
    average: Optional[str] = "macro",
    eval_feature_extraction: bool = False,
):
    assert task != "binary" and num_classes is not None, "For multiclass tasks provide number of classes"

    metrics: dict[str, Callable] = {
        "roc_auc": AUROC(
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "ap": AveragePrecision(
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "accuracy": Accuracy(
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "specificity": Specificity(
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "sensitivity": Recall( # or Sensitivity
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "f1": F1Score(
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "ppv": Precision( # or PPV
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "npv": NegativePredictiveValue(
            task=task,
            num_classes=num_classes,
            average=average
        ),
    }

    if eval_feature_extraction:
        for unsupported_metric in ("ap", "roc_auc"):
            _ = metrics.pop(unsupported_metric, None)

    return metrics


def load_and_validate_dataset(dataset_name: str, eval_args: EvaluateArguments, dataset_split: Optional[str] = None) -> Dataset:
    if os.path.exists(dataset_name):
        # Load from local path
        logger.info(f"Loading dataset {dataset_name} from local path")
        dataset = load_from_disk(
            dataset_path=dataset_name
        )
    elif dataset_name.startswith(DATASET_SCRATCH_PREFIX):
        # Load from scratch directory on Metacentrum
        logger.info(f"Loading dataset {dataset_name} from scratch storage")
        dataset = load_dataset_from_scratch(
            dataset_name
        )
    else:
        # Pull from HF or load it from cache
        logger.info(f"Loading dataset {dataset_name} from HF Hub")
        dataset = load_dataset(
            dataset_name,
            split=dataset_split
        )
    if not isinstance(dataset, Dataset):
        assert dataset_split is not None, "For datasets of type `DatasetDict` you must specify splits to use"
        dataset: Dataset = concatenate_datasets(
            [
                dataset[split]
                for split in dataset_split.split("+")
            ]
        )

    # Validate dataset (check required columns)
    if eval_args.image_column_name not in dataset.column_names:
        raise ValueError(
            f"--image_column_name {eval_args.image_column_name} not found in dataset '{dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct image column - one of "
            f"{', '.join(dataset.column_names)}."
        )

    if eval_args.id_column_name not in dataset.column_names:
        raise ValueError(
            f"--id_column_name {eval_args.id_column_name} not found in dataset '{dataset_name}'. "
            "Make sure to set `--id_column_name` to the correct ID column - one of "
            f"{', '.join(dataset.column_names)}."
        )

    return dataset


def get_labels(dataset: Dataset, model: MelaNet, eval_args: EvaluateArguments) -> Optional[list[str]]:
    label_column = eval_args.label_column_name
    if label_column is not None:
        if label_column not in dataset.column_names:
            raise ValueError(
                f"--label_column_name {label_column} not found in dataset '{eval_args.dataset_name}'. "
                "Make sure to set `--label_column_name` to the correct text column - one of "
                f"{', '.join(dataset.column_names)} - or remove it and skip metrics computations (no ground truth)."
            )

        labels = dataset.features[label_column].names
    else:
        labels = model.get_labels()
    return labels


def has_ground_truth_labels(dataset: Dataset, eval_args: EvaluateArguments) -> bool:
    label_column = eval_args.label_column_name
    return True if label_column is not None and label_column in dataset.column_names else False


def metrics_update_state(
    predictions,
    label_ids,
    metrics: dict[str, Callable],
    classification_task: Literal["binary", "multiclass", "multilabel"],
    predictions_type: Literal["logits", "probs", "labels"] = "logits"
) -> None:
    """Computes accuracy on a batch of predictions"""
    target = torch.tensor(label_ids)
    t_predictions = torch.tensor(predictions)
    if predictions_type == "logits":
        preds = F.softmax(t_predictions, dim=-1) if classification_task == "multiclass" else F.sigmoid(t_predictions)
    elif predictions_type == "probs":
        preds = t_predictions
    elif predictions_type == "labels":
        preds = t_predictions
    else:
        ValueError(f"Unsupported prediction type of '{predictions_type}'. Must be one of ('logits', 'probs', 'labels').")

    for metric_name, metric in metrics.items():
        _res = metric(
            preds=preds,
            target=target
        )


def get_transforms(load_transforms: str) -> list[Callable[[torch.Tensor], torch.Tensor]]:
    _transforms_by_name = build_view_transformations()

    if load_transforms == "all":
        return list(_transforms_by_name.values())
    else:
        transform_names = [name.strip() for name in load_transforms.split(",")]
        return [
            _transforms_by_name[t_name]
            for t_name in transform_names
        ]


def classifier_predict(
    model: MelaNet,
    dataset: Dataset,
    eval_args: EvaluateArguments,
    metrics: Optional[dict] = None,
    tta_transforms: Optional[list[Callable[[torch.Tensor], torch.Tensor]]] = None
) -> dict[str, int]:
    if eval_args.load_cached_model_inference and os.path.isfile(eval_args.load_cached_model_inference):
        cached_values = CacheManager[ClassifierCache].load(
            path=eval_args.load_cached_model_inference
        )
        predictions = cached_values.cache["predictions"]
    else:
        predictions = run_inference(
            model=model,
            dataset=dataset,
            image_column_name=eval_args.image_column_name,
            id_column_name=eval_args.id_column_name,
            model_kwargs={"return_logits": True},
            batch_size=eval_args.batch_size,
            transforms=tta_transforms,
            num_workers=eval_args.num_workers,
            pin_memory=True,
            device_type=eval_args.device
        )

        if eval_args.cache_model_inference:
            CacheManager.save(
                path=eval_args.cache_model_inference,
                cache={
                    "predictions": predictions
                },
                metadata={
                    "eval_dataset": {
                        "name": eval_args.dataset_name,
                        "split": eval_args.eval_split,
                        "size": None
                    },
                    "model": eval_args.model_name_or_path,
                    "classification_task": eval_args.classification_task,
                    "is_feature_extractor": eval_args.eval_as_feature_extraction,
                }
            )

    final_preds = {}
    final_probs = {}
    for lesion_id, pred_logits in predictions.items():
        pred_probs = softmax(pred_logits, axis=-1)
        if eval_args.prediction_resolution_strategy == "greedy":
            sample_type_id, prediction = np.unravel_index(pred_probs.argmax(), pred_probs.shape)
            _probs = pred_probs[sample_type_id]
        elif eval_args.prediction_resolution_strategy == "first":
            prediction = pred_probs[0].argmax()
            _probs = pred_probs[0]
        elif eval_args.prediction_resolution_strategy == "last":
            prediction = pred_probs[-1].argmax()
            _probs = pred_probs[-1]
        elif eval_args.prediction_resolution_strategy == "mean":
            mean_probs = np.mean(pred_probs, axis=0)
            prediction = mean_probs.argmax()
            _probs = mean_probs
        else:
            prediction = -1
            _probs = np.array([])
        final_preds[lesion_id] = int(prediction)
        if metrics is not None:
            final_probs[lesion_id] = _probs

    if metrics is not None:
        logger.info("Computing metrics")
        mapped_labels = pd.DataFrame({
            eval_args.id_column_name: dataset[eval_args.id_column_name],
            eval_args.label_column_name: dataset[eval_args.label_column_name],
        })
        mapped_labels = mapped_labels.drop_duplicates(eval_args.id_column_name)

        if len(mapped_labels) != len(final_preds):
            return final_preds
        mapped_labels["prediction"] = mapped_labels[eval_args.id_column_name].map(final_preds)
        if mapped_labels["prediction"].hasnans:
            return final_preds
        mapped_labels["prediction"] = mapped_labels["prediction"].astype(int)
        mapped_labels["prob"] = mapped_labels[eval_args.id_column_name].map(final_probs)

        metrics_update_state(
            predictions=np.asarray(mapped_labels["prob"].to_list()),
            label_ids=mapped_labels[eval_args.label_column_name].to_list(),
            metrics=metrics,
            classification_task=eval_args.classification_task,
            predictions_type="probs"
        )

    return final_preds


def feature_extraction_predict(
    model: MelaNet,
    index: Dataset,
    dataset: Dataset,
    eval_args: EvaluateArguments,
    metrics: Optional[dict] = None,
    index_transforms: Optional[list[Callable[[torch.Tensor], torch.Tensor]]] = None,
    tta_transforms: Optional[list[Callable[[torch.Tensor], torch.Tensor]]] = None
) -> dict[str, int]:
    if eval_args.load_cached_model_inference and os.path.isfile(eval_args.load_cached_model_inference):
        cached_values = CacheManager[FeatureExtractorCache].load(
            path=eval_args.load_cached_model_inference
        )
        features_columns = list(cached_values.cache["index"].keys())
        for ft_col in features_columns:
            index = index.add_column(ft_col, cached_values.cache["index"][ft_col])
            dataset = dataset.add_column(ft_col, cached_values.cache["eval_dataset"][ft_col])
    else:
        index_extracted_features = run_inference(
            model=model,
            dataset=index,
            image_column_name=eval_args.image_column_name,
            id_column_name=eval_args.id_column_name,
            batch_size=eval_args.batch_size,
            transforms=index_transforms,
            num_workers=eval_args.num_workers,
            pin_memory=True,
            device_type=eval_args.device
        )
        eval_extracted_features = run_inference(
            model=model,
            dataset=dataset,
            image_column_name=eval_args.image_column_name,
            id_column_name=eval_args.id_column_name,
            batch_size=eval_args.batch_size,
            transforms=tta_transforms,
            num_workers=eval_args.num_workers,
            pin_memory=True,
            device_type=eval_args.device
        )

        if model.feature_extractor_config.output_type == "object":
            index_extracted_features = pd.concat(index_extracted_features.values(), ignore_index=True)
            eval_extracted_features = pd.concat(eval_extracted_features.values(), ignore_index=True)

            _feature_names = FeatureExtractorOutput.get_feature_names()
            _to_drop = []
            _to_rename = []
            for f_name in _feature_names:
                if index_extracted_features[f_name].hasnans or eval_extracted_features[f_name].hasnans:
                    _to_drop.append(f_name)
                else:
                    _to_rename.append(f_name)

            if _to_drop:
                index_extracted_features = index_extracted_features.drop(columns=_to_drop)
                eval_extracted_features = eval_extracted_features.drop(columns=_to_drop)
            if _to_rename:
                index_extracted_features = index_extracted_features.rename(
                    columns={f_name: f"{MELANET_FEATURES_PREFIX}_{f_name}" for f_name in _to_rename}
                )
                eval_extracted_features = eval_extracted_features.rename(
                    columns={f_name: f"{MELANET_FEATURES_PREFIX}_{f_name}" for f_name in _to_rename}
                )
        else:
            index_extracted_features = pd.Series(
                index_extracted_features
            ).explode().reset_index(name=MELANET_FEATURES_PREFIX).rename(columns={"index": eval_args.id_column_name})
            eval_extracted_features = pd.Series(
                eval_extracted_features
            ).explode().reset_index(name=MELANET_FEATURES_PREFIX).rename(columns={"index": eval_args.id_column_name})

        index_extracted_features[eval_args.label_column_name] = index_extracted_features[eval_args.id_column_name].map({
            id: label
            for id, label in zip(index[eval_args.id_column_name], index[eval_args.label_column_name])
        })

        features_columns = [col for col in index_extracted_features.columns if col.startswith(MELANET_FEATURES_PREFIX)]
        assert features_columns, "Can not find extracted features"

        if eval_args.cache_model_inference:
            CacheManager.save(
                path=eval_args.cache_model_inference,
                cache={
                    "index": {
                        ft_col: index_extracted_features[ft_col].to_list()
                        for ft_col in features_columns
                    },
                    "eval_dataset": {
                        ft_col: eval_extracted_features[ft_col].to_list()
                        for ft_col in features_columns
                    },
                },
                metadata={
                    "datasets": {
                        "index": {
                            "name": eval_args.index_name,
                            "split": eval_args.index_split,
                            "size": eval_args.index_size
                        },
                        "eval_dataset": {
                            "name": eval_args.dataset_name,
                            "split": eval_args.eval_split,
                            "size": None
                        }
                    },
                    "models": {
                        "ft_model": eval_args.model_name_or_path,
                        "zero_shot_model": eval_args.model_name_or_path,
                        "feature_extractor_config": kwargs2cli(**model.feature_extractor_config.model_dump()),
                        "zero_shot_config": kwargs2cli(**model.zero_shot_config.model_dump()),
                        "classification_task": eval_args.classification_task,
                        "is_feature_extractor": eval_args.eval_as_feature_extraction,
                    }
                }
            )

    vector_store_config = VectorStoreConfig.from_cli(eval_args.vector_store_config) if eval_args.vector_store_config is not None else create_default_vector_store_config()
    logger.info(f"Vector store config {vector_store_config}")
    if len(features_columns) > 1:
        vector_store = MultiNNClassifier(
            cls_ids=np.asarray(index_extracted_features[eval_args.label_column_name].to_list()),
            embeddings={
                feat_name: np.asarray(index_extracted_features[feat_name].to_list(), dtype=np.float32).squeeze()
                for feat_name in features_columns
            },
            metric=vector_store_config.metric,
            pca_components=vector_store_config.pca_components
        )
        indexes_preds_w_scores = vector_store.predict(
            query_embeddings={
                feat_name: np.asarray(eval_extracted_features[feat_name].to_list(), dtype=np.float32).squeeze()
                for feat_name in features_columns
            },
            top_k=vector_store_config.top_k,
            search_k=vector_store_config.search_k,
            unique_only=vector_store_config.unique_only,
            return_scores=True
        )

        if vector_store_config.top_k == 1:
            mapped_predictions = pd.DataFrame(
                [
                    (id_, label, dist, index_name)
                    for index_name, retrieved_labels_w_scores in indexes_preds_w_scores.items() for id_, (label, dist) in zip(eval_extracted_features[eval_args.id_column_name], retrieved_labels_w_scores)
                ],
                columns=[eval_args.id_column_name, "label", "score", "index_name"]
            )
        else:
            mapped_predictions = pd.DataFrame(
                [
                    (id_, *zip(*preds_w_scores), index_name)
                    for index_name, retrieved_labels_w_scores in indexes_preds_w_scores.items() for id_, preds_w_scores in zip(eval_extracted_features[eval_args.id_column_name], retrieved_labels_w_scores)
                ],
                columns=[eval_args.id_column_name, "label", "score", "index_name"]
            )
    else:
        vector_store = NNClassifier(
            cls_ids=np.asarray(index_extracted_features[eval_args.label_column_name].to_list()),
            embeddings=np.asarray(index_extracted_features[features_columns[0]].to_list(), dtype=np.float32).squeeze(),
            metric=vector_store_config.metric,
            pca_components=vector_store_config.pca_components
        )
        retrieved_labels_w_scores = vector_store.predict(
            query_embeddings= np.asarray(eval_extracted_features[features_columns[0]].to_list(), dtype=np.float32).squeeze(),
            top_k=vector_store_config.top_k,
            search_k=vector_store_config.search_k,
            unique_only=vector_store_config.unique_only,
            return_scores=True
        )

        if vector_store_config.top_k == 1:
            mapped_predictions = pd.DataFrame(
                [(id_, label, dist) for id_, (label, dist) in zip(eval_extracted_features[eval_args.id_column_name], retrieved_labels_w_scores)],
                columns=[eval_args.id_column_name, "label", "score"]
            )
        else:
            mapped_predictions = pd.DataFrame(
                [(id_, *zip(*preds_w_scores)) for id_, preds_w_scores in zip(eval_extracted_features[eval_args.id_column_name], retrieved_labels_w_scores)],
                columns=[eval_args.id_column_name, "label", "score"]
            )

    predictions = {}
    if vector_store_config.top_k == 1:
        mapped_predictions_grouped = mapped_predictions.groupby(eval_args.id_column_name)
        for lesion_id in mapped_predictions_grouped.groups:
            lesion_preds = mapped_predictions_grouped.get_group(lesion_id)
            if eval_args.prediction_resolution_strategy == "greedy":
                idx_pred = lesion_preds["score"].argmin() if vector_store_config.metric == "l2" else lesion_preds["score"].argmax()
                prediction = lesion_preds["label"].iloc[idx_pred]
            elif eval_args.prediction_resolution_strategy == "first":
                prediction = lesion_preds["label"].iloc[0]
            elif eval_args.prediction_resolution_strategy == "last":
                prediction = lesion_preds["label"].iloc[-1]
            elif eval_args.prediction_resolution_strategy == "rrf":
                # majority voting
                sorted_lesion_preds = lesion_preds.sort_values(
                    "score",
                    ascending=vector_store_config.metric == "l2"
                )
                rrf_preds = reciprocal_rank_fusion(
                    results=[sorted_lesion_preds["label"].to_list(),],
                    top_n=vector_store_config.rerank_top_n,
                    k=vector_store_config.rrf_k,
                    items_have_scores=False
                )
                prediction = rrf_preds[0]
            else:
                prediction = -1
            predictions[lesion_id] = int(prediction)
    else:
        mapped_predictions_grouped = mapped_predictions.groupby(eval_args.id_column_name)
        for lesion_id in mapped_predictions_grouped.groups:
            lesion_preds = mapped_predictions_grouped.get_group(lesion_id)
            if eval_args.prediction_resolution_strategy == "greedy":
                exploded_lesion_preds = lesion_preds.explode(["label", "score"])
                idx_pred = exploded_lesion_preds["score"].argmin() if vector_store_config.metric == "l2" else exploded_lesion_preds["score"].argmax()
                prediction = exploded_lesion_preds["label"].iloc[idx_pred]
            elif eval_args.prediction_resolution_strategy == "first":
                prediction = lesion_preds["label"].iloc[0][0]
            elif eval_args.prediction_resolution_strategy == "last":
                prediction = lesion_preds["label"].iloc[-1][0]
            elif eval_args.prediction_resolution_strategy == "rrf":
                rrf_preds = reciprocal_rank_fusion(
                    results=lesion_preds["label"].to_list(),
                    top_n=vector_store_config.rerank_top_n,
                    k=vector_store_config.rrf_k,
                    items_have_scores=False
                )
                prediction = rrf_preds[0]
            else:
                prediction = -1
            predictions[lesion_id] = int(prediction)

    if metrics is not None:
        logger.info("Computing metrics")
        mapped_labels = pd.DataFrame({
            eval_args.id_column_name: dataset[eval_args.id_column_name],
            eval_args.label_column_name: dataset[eval_args.label_column_name],
        })
        mapped_labels = mapped_labels.drop_duplicates(eval_args.id_column_name)

        if len(mapped_labels) != len(predictions):
            return predictions
        mapped_labels["prediction"] = mapped_labels[eval_args.id_column_name].map(predictions)
        if mapped_labels["prediction"].hasnans:
            return predictions
        mapped_labels["prediction"] = mapped_labels["prediction"].astype(int)

        metrics_update_state(
            predictions=np.asarray(mapped_labels["prediction"].to_list()),
            label_ids=mapped_labels[eval_args.label_column_name].to_list(),
            metrics=metrics,
            classification_task=eval_args.classification_task,
            predictions_type="labels"
        )

    return predictions


def evaluate(eval_args: EvaluateArguments):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = (eval_args.log_level if eval_args.log_level else eval_args.default_log_level).upper()
    if log_level not in log_levels:
        log_level = eval_args.default_log_level.upper()
    logger.setLevel(log_level)

    logger.info(f"Evaluation parameters {eval_args}")

    # Prepare and load dataset
    if eval_args.eval_as_feature_extraction:
        assert eval_args.index_name is not None, "For zero-shot classification provide dataset `index_name` that will be used as index"
        assert eval_args.label_column_name is not None, "When using feature extractor then index must include class labels"

        index = load_and_validate_dataset(
            dataset_name=eval_args.index_name,
            eval_args=eval_args,
            dataset_split=eval_args.index_split
        )
        dataset = load_and_validate_dataset(
            dataset_name=eval_args.dataset_name,
            eval_args=eval_args,
            dataset_split=eval_args.eval_split
        )
        logger.info(f"Loaded index {index}")
        logger.info(f"Loaded eval dataset {dataset}")
        if eval_args.index_size is not None:
            _index_size = eval_args.index_size if eval_args.index_size < 1.0 else int(eval_args.index_size)
            if isinstance(_index_size, int):
                assert _index_size < len(index), "Specify number of samples that is lower then size of the dataset"
                split_perc = _index_size / len(index)
            elif isinstance(_index_size, float):
                assert 0.0 < _index_size < 1.0, "Specify portion of dataset to use by value between 0.0 and 1.0"
                split_perc = _index_size
            else:
                raise ValueError("If specifying index size to use then provide `int` or `float` value")

            index = index.train_test_split(
                train_size=split_perc,
                shuffle=True,
                stratify_by_column=eval_args.label_column_name,
                seed=eval_args.seed
            )["train"]
    else:
        index = None
        dataset = load_and_validate_dataset(
            dataset_name=eval_args.dataset_name,
            eval_args=eval_args,
            dataset_split=eval_args.eval_split
        )
        logger.info(f"Loaded eval dataset {dataset}")

    # Init model
    model = MelaNet(
        model_name=eval_args.model_name_or_path,
        is_feature_extractor=eval_args.eval_as_feature_extraction,
        feature_extractor_config=FeatureExtractorConfig.from_cli(eval_args.feature_extractor_config) if eval_args.feature_extractor_config is not None else None,
        zero_shot_model_name=eval_args.zero_shot_model_name_or_path,
        zero_shot_config=ZeroShotConfig.from_cli(eval_args.zero_shot_config) if eval_args.zero_shot_config is not None else None,
        device=eval_args.device
    )

    labels = get_labels(
        dataset=index if eval_args.eval_as_feature_extraction else dataset,
        model=model,
        eval_args=eval_args
    )
    has_gt_labels = has_ground_truth_labels(
        dataset=dataset,
        eval_args=eval_args
    )
    if has_gt_labels:
        logger.info(f"Loaded eval dataset has {len(labels)} ground truth labels")

    # Load metrics for given task
    metrics = _get_classification_metrics(
        task=eval_args.classification_task,
        num_classes=len(labels),
        eval_feature_extraction=eval_args.eval_as_feature_extraction
    ) if labels is not None and has_gt_labels else None
    logger.info(f"Using metrics {metrics}")

    # Load augmentations
    index_transforms = None
    tta_transforms = None
    if eval_args.apply_augmentations:
        __view_transforms = get_transforms(eval_args.apply_augmentations)
        index_transforms = __view_transforms
        tta_transforms = __view_transforms
    else:
        if eval_args.index_augmentations:
            index_transforms = get_transforms(eval_args.index_augmentations)
        if eval_args.test_time_augmentations:
            tta_transforms = get_transforms(eval_args.test_time_augmentations)

    # Run evaluation
    if eval_args.eval_as_feature_extraction:
        logger.info("Running eval of feature extractor")
        predictions = feature_extraction_predict(
            model=model,
            index=index,
            dataset=dataset,
            eval_args=eval_args,
            metrics=metrics,
            index_transforms=index_transforms,
            tta_transforms=tta_transforms
        )
    else:
        logger.info("Running eval of classifier")
        predictions = classifier_predict(
            model=model,
            dataset=dataset,
            eval_args=eval_args,
            metrics=metrics,
            tta_transforms=tta_transforms
        )

    results = {
        "datetime": datetime.now().isoformat(),
        "model": eval_args.model_name_or_path,
        "zero_shot_model": eval_args.zero_shot_model_name_or_path,
        "dataset": {
            "name": eval_args.dataset_name,
            "split": eval_args.eval_split
        },
        "task": eval_args.classification_task,
        "as_feature_extractor": eval_args.eval_as_feature_extraction,
        "prediction_resolution_strategy": eval_args.prediction_resolution_strategy,
        "predictions": predictions,
    }
    if metrics is not None:
        eval_metric = {
            metric_name: tensor2value(metric.compute().float())
            for metric_name, metric in metrics.items()
        }
        results["metrics"] = eval_metric
    if eval_args.eval_as_feature_extraction:
        results["index"] = {
            "name": eval_args.index_name,
            "split": eval_args.index_split,
            "size": eval_args.index_size
        }
        results["feature_extractor_config"] = eval_args.feature_extractor_config
        results["zero_shot_config"] = eval_args.zero_shot_config
        results["vector_store_config"] = eval_args.vector_store_config
        if eval_args.apply_augmentations:
            results["index_augmentations"] = eval_args.apply_augmentations
            results["test_time_augmentations"] = eval_args.apply_augmentations
        else:
            results["index_augmentations"] = eval_args.index_augmentations
            results["test_time_augmentations"] = eval_args.test_time_augmentations

    logger.info(f"Saving final results to file {eval_args.results_path}")
    with open(eval_args.results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if eval_args.isic_submission_path and eval_args.isic_submission_path.endswith(".csv"):
        try:
            # Currently set up for MILK10k dataset
            logger.info(f"Converting final results to ISIC/MILK submission CSV {eval_args.isic_submission_path}")
            format_isic_submission(
                predictions=predictions,
                labels=labels,
                dataset=dataset,
                image_id_column_name="isic_id",
                lesion_id_column_name=eval_args.id_column_name,
                csv_submission_path=eval_args.isic_submission_path,
                drop_duplicates=True
            )
        except AssertionError as a_e:
            logger.warning(a_e)
        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    parser = HfArgumentParser(EvaluateArguments)
    eval_args = cast(
        EvaluateArguments,
        parser.parse_args_into_dataclasses()[0]
    )

    evaluate(eval_args=eval_args)