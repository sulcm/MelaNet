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

from melanet import MelaNet
from metacentrum_utils import DATASET_SCRATCH_PREFIX, load_dataset_from_scratch


logger = logging.getLogger(__name__)


@dataclass
class EvaluateArguments:
    """
    Arguments pertaining to how evaluate given model on dataset
    """

    classification_task: Literal["binary", "multiclass", "multilabel"] = field(
        metadata={"help": "Specify classification task / objective. Possible values are 'binary', 'multiclass', or 'multilabel'."}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to finetuned model or model identifier from huggingface.co/models"},
    )
    dataset_name: str = field(
        metadata={"help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."},
    )
    results_path: str = field(
        metadata={"help": "Path and file where to save results. Must be JSON file."}
    )
    dataset_split: Optional[str] = field(
        default=None,
        metadata={
            "help": "If `None` then assume its single dataset of type `Dataset` otherwise provide name of the split or splits (e.g. 'test', 'validation+test', etc.)."
        }
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


def metrics_update_state(predictions, label_ids, metrics: dict[str, Callable]):
    """Computes accuracy on a batch of predictions"""
    logits = torch.tensor(predictions)
    target = torch.tensor(label_ids)
    # preds = torch.argmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)

    for metric_name, metric in metrics.items():
        _res = metric(
            preds=probs,
            target=target
        )


def tensor_to_value(x: torch.Tensor):
    return x.item() if x.numel() == 1 else x.tolist()


def evaluate(eval_args: EvaluateArguments):
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
            split=eval_args.dataset_split
        )
    if not isinstance(dataset, Dataset):
        dataset: Dataset = concatenate_datasets(
            [
                dataset[split]
                for split in eval_args.dataset_split.split("+")
            ]
        )

    if eval_args.image_column_name not in dataset.column_names:
        raise ValueError(
            f"--image_column_name {eval_args.image_column_name} not found in dataset '{eval_args.dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct image column - one of "
            f"{', '.join(dataset.column_names)}."
        )

    labels: Optional[list] = None
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

    metrics = _get_classification_metrics(
        task=eval_args.classification_task,
        num_classes=len(labels)
    ) if labels is not None else None

    model = MelaNet(
        model_name=eval_args.model_name_or_path,
        device=eval_args.device
    )

    def map_eval(batch):
        predictions = model(batch[eval_args.image_column_name], return_logits=True)
        batch["predictions"] = predictions
        if metrics is not None:
            metrics_update_state(
                predictions=predictions,
                label_ids=batch[label_column],
                metrics=metrics
            )
        return batch

    dataset = dataset.map(map_eval, batched=True, batch_size=eval_args.batch_size, desc="Evaluating model", load_from_cache_file=False)
    predictions = np.argmax(dataset["predictions"], axis=-1)
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
            metric_name: tensor_to_value(metric.compute().float())
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