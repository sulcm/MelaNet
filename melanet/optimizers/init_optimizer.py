# Implementation based on `transformers.trainer.Trainer`
# GitHub Link: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py

import torch.nn as nn

from typing import Optional, Any

from torch.optim import Optimizer
from transformers.trainer import TrainingArguments, Trainer, get_parameter_names


def init_optimizer(
    model: nn.Module,
    training_args: TrainingArguments,
    optimizer_cls_and_kwargs: Optional[tuple[type[Optimizer], dict[str, Any]]] = None
) -> Optimizer:
    decay_parameters = get_decay_parameter_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if optimizer_cls_and_kwargs is not None:
        optimizer_cls, optimizer_kwargs = optimizer_cls_and_kwargs
    else:
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args, model)

    # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
    # e.g. for GaLore optimizer.
    if "params" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("params")

    # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
    # e.g. for LOMO optimizer.
    if "model" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("model")

    # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
    # to avoid arguments conflicts.
    if "optimizer_dict" in optimizer_kwargs:
        optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


def get_decay_parameter_names(model: nn.Module) -> list[str]:
    """
    Get all parameter names that weight decay will be applied to.

    This function filters out parameters in two ways:
    1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
    2. By parameter name patterns (containing 'bias', or variation of 'norm')
    """
    forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
    return decay_parameters