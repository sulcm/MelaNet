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
from melanet.embeddings.types import MELANET_FEATURES_PREFIX
from melanet.vectorstores import VectorStoreConfig, MultiNNClassifier, NNClassifier
from melanet.vectorstores.rrf import melanet_rrf
from melanet.utils import tensor2value
from metacentrum_utils import DATASET_SCRATCH_PREFIX, load_dataset_from_scratch


logger = logging.getLogger(__name__)


@dataclass
class EvaluateArguments:
    """
    Arguments pertaining to how evaluate given model on dataset
    """

    classification_task: Literal["binary", "multiclass", "multilabel", "feature_classification"] = field(
        metadata={"help": "Specify classification task / objective. Possible values are 'binary', 'multiclass', 'multilabel', or 'feature_classification'."}
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
    eval_split: Optional[str] = field(
        default=None,
        metadata={
            "help": "If `None` then assume its single dataset of type `Dataset` otherwise provide name of the split or splits (e.g. 'test', 'validation+test', etc.)."
        }
    )
    index_split: Optional[str] = field(
        default=None,
        metadata={"help": "Used for `feature_classification` to build index."}
    )
    feature_extractor_config: Optional[str] = field(
        default=None,
        metadata={"help": "Used for configuring feature extraction process."}
    )
    zero_shot_config: Optional[str] = field(
        default=None,
        metadata={"help": "Used for configuring zero-shot process."}
    )
    vector_store_config: Optional[str] = field(
        default=None,
        metadata={"help": "Used for configuring vector stores."}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "The batch size used for inference"}
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


def _get_classification_metrics(
    task: Literal["binary", "multiclass", "multilabel"],
    num_classes: Optional[int] = None,
    average: Optional[str] = "macro"
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
    return metrics


def load_and_validate_dataset(eval_args: EvaluateArguments, dataset_split: Optional[str] = None) -> Dataset:
    if os.path.exists(eval_args.dataset_name):
        # Load from local path
        logger.info(f"Loading dataset {eval_args.dataset_name} from local path")
        dataset = load_from_disk(
            dataset_path=eval_args.dataset_name
        )
    elif eval_args.dataset_name.startswith(DATASET_SCRATCH_PREFIX):
        # Load from scratch directory on Metacentrum
        logger.info(f"Loading dataset {eval_args.dataset_name} from scratch storage")
        dataset = load_dataset_from_scratch(
            eval_args.dataset_name
        )
    else:
        # Pull from HF or load it from cache
        logger.info(f"Loading dataset {eval_args.dataset_name} from HF Hub")
        dataset = load_dataset(
            eval_args.dataset_name,
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
            f"--image_column_name {eval_args.image_column_name} not found in dataset '{eval_args.dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct image column - one of "
            f"{', '.join(dataset.column_names)}."
        )

    return dataset


def get_labels_from_dataset(dataset: Dataset, eval_args: EvaluateArguments) -> Optional[list]:
    labels = None
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
        labels = None
    return labels


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


def classifier_predict(model: MelaNet, dataset: Dataset, eval_args: EvaluateArguments, metrics: Optional[dict] = None) -> np.ndarray:
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

    dataset = dataset.map(map_eval, batched=True, batch_size=eval_args.batch_size, desc="Evaluating model", load_from_cache_file=False)
    predictions = np.argmax(dataset["predictions"], axis=-1)
    return predictions


def feature_extraction_predict(model: MelaNet, index: Dataset, dataset: Dataset, eval_args: EvaluateArguments, metrics: Optional[dict] = None) -> np.ndarray:
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

    index = index.map(map_feature_extraction, batched=True, batch_size=eval_args.batch_size, desc="Extractiong features for index", load_from_cache_file=False)
    dataset = dataset.map(map_feature_extraction, batched=True, batch_size=eval_args.batch_size, desc="Extractiong features for eval dataset", load_from_cache_file=False)

    features_columns = [col for col in index.column_names if col.startswith(MELANET_FEATURES_PREFIX)]
    assert features_columns, "Can not find extracted features"

    vector_store_config = VectorStoreConfig.from_cli(eval_args.vector_store_config) if eval_args.vector_store_config is not None else None
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

    return predictions


def evaluate(eval_args: EvaluateArguments):
    is_feature_extractor = eval_args.classification_task == "feature_classification"

    # Prepare and load dataset
    if is_feature_extractor:
        index = load_and_validate_dataset(eval_args=eval_args, dataset_split=eval_args.index_split)
        dataset = load_and_validate_dataset(eval_args=eval_args, dataset_split=eval_args.eval_split)

        labels = get_labels_from_dataset(dataset=dataset, eval_args=eval_args)
    else:
        index = None
        dataset = load_and_validate_dataset(eval_args=eval_args, dataset_split=eval_args.eval_split)

        labels = get_labels_from_dataset(dataset=dataset, eval_args=eval_args)

    # Load metrics for given task
    metrics = _get_classification_metrics(
        task=eval_args.classification_task,
        num_classes=len(labels)
    ) if labels is not None else None

    # Init model
    model = MelaNet(
        model_name=eval_args.model_name_or_path,
        is_feature_extractor=is_feature_extractor,
        feature_extractor_config=FeatureExtractorConfig.from_cli(eval_args.feature_extractor_config) if eval_args.feature_extractor_config is not None else None,
        zero_shot_model_name=eval_args.zero_shot_model_name_or_path,
        zero_shot_config=ZeroShotConfig.from_cli(eval_args.zero_shot_config) if eval_args.zero_shot_config is not None else None,
        device=eval_args.device
    )

    # Run evaluation
    if is_feature_extractor:
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
        "dataset": {
            "name": eval_args.dataset_name,
            "split": eval_args.dataset_split
        },
        "task": eval_args.classification_task,
        "predictions": predictions.tolist(),
    }
    if metrics is not None:
        eval_metric = {
            metric_name: tensor2value(metric.compute().float())
            for metric_name, metric in metrics.items()
        }
        results["metrics"] = eval_metric

    with open(eval_args.results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = HfArgumentParser(EvaluateArguments)
    eval_args = cast(
        EvaluateArguments,
        parser.parse_args_into_dataclasses()[0]
    )

    evaluate(eval_args=eval_args)