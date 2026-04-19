import os
import torch
import torch.nn as nn
import safetensors.torch

from typing import Optional, Union, Callable, Any

from torch.optim import Optimizer
from transformers import Trainer, TrainingArguments, set_seed
from datasets import Dataset

from .base import BaseAdapter
from .config import FeatureAdapterConfig


class LearnableAdapter(nn.Module, BaseAdapter):
    SAFETENSORS_EXT = "safetensors"
    SAFE_WEIGHTS_NAME = "model.safetensors"
    WEIGHTS_NAME = "pytorch_model.bin"

    activation_str2fn: dict[str, nn.Module] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }

    def fit(
        self,
        training_args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        feature_column_names: Union[str, list[str]],
        label_column_name: str,
        optimizer_cls_and_kwargs: tuple[type[Optimizer], dict[str, Any]],
        compute_loss_func: Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], float]],
        compute_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], dict[str, float]]] = None
    ) -> "LearnableAdapter":
        if training_args.seed is not None:
            set_seed(training_args.seed)

        if self.adapter_type == "linear":
            assert isinstance(feature_column_names, str)

            if label_column_name not in train_dataset.column_names:
                def collate_fn(batch) -> tuple[torch.Tensor, None]:
                    features = torch.stack([sample[feature_column_names] for sample in batch])
                    return {"features": features, "labels": None}
            else:
                def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
                    features = torch.stack([sample[feature_column_names] for sample in batch])
                    labels = torch.tensor([sample[label_column_name] for sample in batch])
                    return {"features": features, "labels": labels}
        elif self.adapter_type == "fusion":
            assert isinstance(feature_column_names, list), "When training `fusion` adapter input must be list of Tensors from each backbone network."

            if label_column_name not in train_dataset.column_names:
                def collate_fn(batch) -> tuple[list[torch.Tensor], None]:
                    features = [
                        torch.stack([sample[ft_name] for sample in batch])
                        for ft_name in feature_column_names
                    ]
                    return {"features": features, "labels": None}
            else:
                def collate_fn(batch) -> tuple[list[torch.Tensor], torch.Tensor]:
                    features = [
                        torch.stack([sample[ft_name] for sample in batch])
                        for ft_name in feature_column_names
                    ]
                    labels = torch.tensor([sample[label_column_name] for sample in batch])
                    return {"features": features, "labels": labels}
        else:
            raise ValueError(
                f"Currently unsupported learnable adapter type {self.adapter_type}"
            )

        # Initialize trainer
        if training_args.label_names is None:
            training_args.label_names = ["labels"]

        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            compute_loss_func=compute_loss_func,
            data_collator=collate_fn,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        )

        # Training
        if training_args.do_train:
            train_result = trainer.train()
            trainer.log_metrics("train", train_result.metrics)

        # Evaluation
        if training_args.do_eval:
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)

        self._is_fitted = True
        self.eval()
        return self

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "LearnableAdapter":
        checkpoint = torch.load(pretrained_path)

        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model._is_fitted = True

        return model

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: FeatureAdapterConfig) -> "LearnableAdapter":
        model = cls.from_config(config)

        if os.path.isfile(checkpoint_path):
            _checkpoint_path = checkpoint_path
        elif os.path.isdir(checkpoint_path):
            safe_weights_file = os.path.join(checkpoint_path, cls.SAFE_WEIGHTS_NAME)
            weights_file = os.path.join(checkpoint_path, cls.WEIGHTS_NAME)
            if os.path.isfile(safe_weights_file):
                _checkpoint_path = safe_weights_file
            elif os.path.isfile(weights_file):
                _checkpoint_path = weights_file
            else:
                raise ValueError(
                    f"In checkpoint directory {checkpoint_path} must be file {cls.SAFE_WEIGHTS_NAME} or {cls.WEIGHTS_NAME}"
                )
        else:
            raise ValueError(
                f"Checkpoint must be file or directory including file {cls.SAFE_WEIGHTS_NAME} or {cls.WEIGHTS_NAME}"
            )

        if _checkpoint_path.endswith(f".{cls.SAFETENSORS_EXT}"):
            state_dict = safetensors.torch.load_file(_checkpoint_path)
        else:
            _loaded_checkpoint = torch.load(_checkpoint_path, weights_only=True)
            state_dict = _loaded_checkpoint["state_dict"]

        model.load_state_dict(state_dict, strict=True)
        model._is_fitted = True

        return model