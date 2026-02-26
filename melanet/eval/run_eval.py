import os
import json
import logging
import numpy as np

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

from melanet.melanet_wrapper import MelaNet, FeatureExtractorConfig, ZeroShotConfig
from melanet.cache import CacheManager, ClassifierCache, FeatureExtractorCache
from melanet.embeddings.types import MELANET_FEATURES_PREFIX
from melanet.vectorstores import VectorStoreConfig, create_default_vector_store_config, MultiNNClassifier, NNClassifier
from melanet.vectorstores.rrf import melanet_rrf
from melanet.utils import tensor2value
from formatter.isic import format_isic_submission
from metacentrum_utils import DATASET_SCRATCH_PREFIX, load_dataset_from_scratch


logger = logging.getLogger(__name__)


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
    device: str = field(
        default="cuda",
        metadata={"help": "Select which device to use for inference."}
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed that will be set to ensure reproducibility. Defaults to 42."}
    )


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


def metrics_update_state(predictions, label_ids, metrics: dict[str, Callable], predictions_type: Literal["logits", "probs", "labels"] = "logits"):
    """Computes accuracy on a batch of predictions"""
    target = torch.tensor(label_ids)
    t_predictions = torch.tensor(predictions)
    if predictions_type == "logits":
        probs = F.softmax(t_predictions, dim=-1)
        preds = probs.argmax(dim=-1)
    elif predictions_type == "probs":
        preds = t_predictions.argmax(dim=-1)
    elif predictions_type == "labels":
        preds = t_predictions
    else:
        ValueError(f"Unsupported prediction type of '{predictions_type}'. Must be one of ('logits', 'probs', 'labels').")

    for metric_name, metric in metrics.items():
        _res = metric(
            preds=preds,
            target=target
        )


def classifier_predict(model: MelaNet, dataset: Dataset, eval_args: EvaluateArguments, metrics: Optional[dict] = None) -> list[int]:
    def map_eval(batch):
        with torch.no_grad():
            predictions = model(
                batch[eval_args.image_column_name],
                return_logits=True
            )
        batch["predictions"] = predictions
        if metrics is not None:
            metrics_update_state(
                predictions=predictions,
                label_ids=batch[eval_args.label_column_name],
                metrics=metrics,
                predictions_type="logits"
            )
        return batch

    if eval_args.load_cached_model_inference and os.path.isfile(eval_args.load_cached_model_inference):
        cached_values = CacheManager[ClassifierCache].load(
            path=eval_args.load_cached_model_inference
        )
        logits = cached_values.cache["logits"]
    else:
        dataset = dataset.map(map_eval, batched=True, batch_size=eval_args.batch_size, desc="Evaluating model", load_from_cache_file=False)
        logits = np.array(dataset["predictions"])

        if eval_args.cache_model_inference:
            CacheManager.save(
                path=eval_args.cache_model_inference,
                cache={
                    "logits": logits
                },
                metadata={
                    "eval_dataset": {
                        "name": eval_args.dataset_name,
                        "split": eval_args.eval_split,
                        "size": None
                    },
                    "model": eval_args.model_name_or_path
                }
            )

    predictions = np.argmax(logits, axis=-1)
    return predictions.tolist()


def feature_extraction_predict(model: MelaNet, index: Dataset, dataset: Dataset, eval_args: EvaluateArguments, metrics: Optional[dict] = None) -> list[int]:
    is_model_output_object = model.feature_extractor_config.output_type == "object"

    def map_feature_extraction(batch):
        with torch.no_grad():
            features = model(
                batch[eval_args.image_column_name]
            )
        if is_model_output_object:
            features = features.to_dict()
            for feat_name in features.keys():
                if features[feat_name] is not None:
                    batch[MELANET_FEATURES_PREFIX + "_" + feat_name] = features[feat_name]
        else:
            batch[MELANET_FEATURES_PREFIX + "_embeddings"] = features
        return batch

    if eval_args.load_cached_model_inference and os.path.isfile(eval_args.load_cached_model_inference):
        cached_values = CacheManager[FeatureExtractorCache].load(
            path=eval_args.load_cached_model_inference
        )
        features_columns = list(cached_values.cache["index"].keys())
        for ft_col in features_columns:
            index = index.add_column(ft_col, cached_values.cache["index"][ft_col])
            dataset = dataset.add_column(ft_col, cached_values.cache["eval_dataset"][ft_col])
    else:
        index = index.map(map_feature_extraction, batched=True, batch_size=eval_args.batch_size, desc="Extractiong features for index", load_from_cache_file=False)
        dataset = dataset.map(map_feature_extraction, batched=True, batch_size=eval_args.batch_size, desc="Extractiong features for eval dataset", load_from_cache_file=False)

        features_columns = [col for col in index.column_names if col.startswith(MELANET_FEATURES_PREFIX)]
        assert features_columns, "Can not find extracted features"

        if eval_args.cache_model_inference:
            CacheManager.save(
                path=eval_args.cache_model_inference,
                cache={
                    "index": {
                        ft_col: torch.stack(index[ft_col])
                        for ft_col in features_columns
                    },
                    "eval_dataset": {
                        ft_col: torch.stack(dataset[ft_col])
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
                        "feature_extractor_config": model.feature_extractor_config.model_dump(),
                        "zero_shot_config": model.zero_shot_config.model_dump()
                    }
                }
            )

    vector_store_config = VectorStoreConfig.from_cli(eval_args.vector_store_config) if eval_args.vector_store_config is not None else create_default_vector_store_config()
    if len(features_columns) > 1:
        vector_store = MultiNNClassifier(
            cls_ids=np.asarray(index[eval_args.label_column_name]),
            embeddings={
                feat_name: np.asarray(index[feat_name])
                for feat_name in features_columns
            },
            metric=vector_store_config.metric,
            pca_components=vector_store_config.pca_components
        )

        indexes_preds = vector_store.predict(
            query_embeddings={
                feat_name: np.asarray(dataset[feat_name])
                for feat_name in features_columns
            },
            top_k=vector_store_config.top_k,
            search_k=vector_store_config.search_k,
            return_distances=vector_store_config.return_distances
        )
        predictions = melanet_rrf(
            retrieved_results=indexes_preds,
            top_n=vector_store_config.rerank_top_n,
            k=vector_store_config.rrf_k
        )
    else:
        vector_store = NNClassifier(
            cls_ids=np.asarray(index[eval_args.label_column_name]),
            embeddings=np.asarray(index[features_columns[0]]),
            metric=vector_store_config.metric,
            pca_components=vector_store_config.pca_components
        )

        predictions = vector_store.predict(
            query_embeddings= np.asarray(dataset[features_columns[0]]),
            top_k=vector_store_config.top_k,
            search_k=vector_store_config.search_k,
            return_distances=vector_store_config.return_distances
        )

    if metrics is not None:
        metrics_update_state(
            predictions=predictions,
            label_ids=dataset[eval_args.label_column_name],
            metrics=metrics,
            predictions_type="labels"
        )

    return predictions.tolist()


def evaluate(eval_args: EvaluateArguments):
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

    # Load metrics for given task
    metrics = _get_classification_metrics(
        task=eval_args.classification_task,
        num_classes=len(labels),
        eval_feature_extraction=eval_args.eval_as_feature_extraction
    ) if labels is not None and has_gt_labels else None

    # Run evaluation
    if eval_args.eval_as_feature_extraction:
        predictions = feature_extraction_predict(
            model=model,
            index=index,
            dataset=dataset,
            eval_args=eval_args,
            metrics=metrics
        )
    else:
        predictions = classifier_predict(
            model=model,
            dataset=dataset,
            eval_args=eval_args,
            metrics=metrics
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
        "predictions": predictions,
    }
    if metrics is not None:
        eval_metric = {
            metric_name: tensor2value(metric.compute().float())
            for metric_name, metric in metrics.items()
        }
        results["metrics"] = eval_metric

    with open(eval_args.results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if eval_args.isic_submission_path and eval_args.isic_submission_path.endswith(".csv"):
        try:
            # Currently set up for MILK10k dataset
            format_isic_submission(
                predictions=predictions,
                labels=labels,
                dataset=dataset,
                image_id_column_name="isic_id",
                lesion_id_column_name="lesion_id",
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