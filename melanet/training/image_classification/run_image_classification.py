#!/usr/bin/env python
"""
Fine-tuning a 🤗 Transformers model for image classification.

Modified version of file located at: https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py
"""

# TODO: This tag signifies change or need for behavior control

import os
import sys
import logging
import transformers
import torch
import evaluate
import numpy as np
import torch.nn.functional as F

from typing import cast, Optional, Callable, Literal
from dataclasses import dataclass, field
from functools import partial
from collections import Counter

from PIL import Image
from datasets import load_dataset, load_from_disk
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    ViTHybridImageProcessor,
    ViTHybridConfig,
    TimmWrapperImageProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version, send_example_telemetry
# from transformers.utils.versions import require_version

from loss_functions import FocalLoss, SupConLoss
from metacentrum_utils import load_dataset_from_scratch, DATASET_SCRATCH_PREFIX


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.57.0.dev0")
# require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."},
    )
    label_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class AuxiliaryArguments:
    """
    Auxiliary arguments to specify/define for model/data/training
    """

    # Loss
    loss_reduction: Literal["sum", "mean", "w_mean"] = field(
        default="sum",
        metadata={"help": (
            "How to combine multiple losses from applied loss functions."
            " - 'sum': Sums all results"
            " - 'mean': Returns the mean of applied loss functions"
            " - 'w_mean': Returns the weighted mean of applied loss functions (weight == loss multiplier)"
            " Defaults to 'sum'."
        )}
    )
    ## CE
    ce_loss_multiplier: Optional[float] = field(
        default=None,
        metadata={"help": (
            "Value that multiplies CE loss. If `None` then do not apply CE loss."
            " If every other loss has multiplier set to `None` then default to `ce_loss_multiplier=1.0`"
        )}
    )
    ## Focal
    focal_loss_multiplier: Optional[float] = field(
        default=None,
        metadata={"help": "Value that multiplies Focal loss. If `None` then do not apply Focal loss."}
    )
    focal_loss_task: Literal["binary", "multiclass", "multilabel"] = field(
        default="multiclass",
        metadata={"help": "Fine-tuning task such as 'binary', 'multiclass', or 'multilabel'"}
    )
    focal_loss_gamma: float = field(
        default=2.0,
        metadata={"help": (
            "Controls how much to down-weight easy examples."
            " Defaults to `2.0`"
            " as in original [paper](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)."
        )}
    )
    focal_loss_alpha: Optional[float] = field(
        default=None,
        metadata={"help": (
            "Balances positive vs. negative classes weights."
            " Enter single weight using floating point number (usually `[0.25, 0.75]`)."
            " If `-1` entered then weights are auto selected. For binary task is alpha set to `0.25` otherwise list of inverse frequencies of classes is used."
            " Defaults to `None`."
        )}
    )
    ## SupCon (Supervised Contrastive)
    supcon_loss_multiplier: Optional[float] = field(
        default=None,
        metadata={"help": "Value that multiplies SupCon loss"}
    )
    supcon_loss_temperature: float = field(
        default=0.07,
        metadata={"help": "Controls how peaked the distribution of similarities is in contrastive learning. Defaults to `0.07` as recommended in paper."}
    )
    supcon_loss_base_temperature: Optional[float] = field(
        default=None,
        metadata={"help": (
            "Normalization constant that balances scaling of the loss"
            " `loss = - (temperature / base_temperature) * mean_log_prob_pos`."
            " Defaults to `None` then is set to the same value as `temperature` parameter."
        )}
    )
    supcon_loss_contrast_mode: Literal["all", "one"] = field(
        default="all",
        metadata={"help": (
            "How to contrast anchor againts other positives samples."
            " - 'all' more memory intensive -> usually better results;"
            " - 'one' less memory usage;"
            " Defaults to 'all'."
        )}
    )
    # Models
    ## Dropouts
    dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Global dropout option. Applied if found in model config. `None` means default (config) value."}
    )
    hidden_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Hidden layers dropout option. Applied if found in model config. `None` means default (config) value."}
    )
    attention_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Attention layer dropout option. Applied if found in model config. `None` means default (config) value."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AuxiliaryArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        parsed_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        parsed_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args, aux_args = cast(tuple[ModelArguments, DataTrainingArguments, TrainingArguments, AuxiliaryArguments], parsed_args)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_image_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        if os.path.exists(data_args.dataset_name):
            dataset = load_from_disk(
                dataset_path=data_args.dataset_name
            )
        elif data_args.dataset_name.startswith(DATASET_SCRATCH_PREFIX):
            dataset = load_dataset_from_scratch(
                data_args.dataset_name
            )
        else:
            dataset = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    if data_args.image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {data_args.image_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct audio column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    if data_args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(dataset_column_names)}."
        )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[data_args.label_column_name] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features[data_args.label_column_name].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # TODO: Add metric comutation
    # Load selected metrics from the datasets package
    metrics: dict[str, Callable] = {
        "f1": partial(evaluate.load("f1", "multiclass").compute, average="macro"),
        "precision": partial(evaluate.load("precision", "multiclass").compute, average="macro"),
        "recall": partial(evaluate.load("recall", "multiclass").compute, average="macro"),
        "accuracy": evaluate.load("accuracy", "multiclass").compute,
        "roc_auc": partial(evaluate.load("roc_auc", "multiclass").compute, average="macro", multi_class="ovr"),
    }

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        preds = np.argmax(p.predictions, axis=-1)
        probs = F.softmax(torch.tensor(p.predictions), dim=-1).numpy()
        refs = p.label_ids

        results = {}
        for metric_name, metric in metrics.items():
            if metric_name == "roc_auc":
                _res = metric(
                    prediction_scores=probs,
                    references=refs,
                )
            else:
                _res = metric(
                    predictions=preds,
                    references=refs
                )
            if _res is not None:
                results.update(_res)
            else:
                results[metric_name] = None
        return results

    # TODO: Define custom loss calculation
    if (
        aux_args.ce_loss_multiplier is None
        and aux_args.focal_loss_multiplier is None
        and aux_args.supcon_loss_multiplier is None
    ):
        logger.warning(
            "Missing specified loss function. Falling back to CE loss with multiplier set to 1.0"
        )
        aux_args.ce_loss_multiplier = 1.0
    if aux_args.focal_loss_alpha == -1.0:
        if aux_args.focal_loss_task == "binary":
            focal_loss_alpha = 0.25
        else:
            frequencies = Counter(dataset["train"][data_args.label_column_name])
            focal_loss_alpha = [
                1.0 / frequencies.get(l_id, 4)
                for l_id, l in enumerate(labels)
            ]
    else:
        focal_loss_alpha = aux_args.focal_loss_alpha
    F_focal_loss = FocalLoss(
        gamma=aux_args.focal_loss_gamma,
        alpha=focal_loss_alpha,
        task_type=aux_args.focal_loss_task
    )
    F_supcon_loss = SupConLoss(
        temperature=aux_args.supcon_loss_temperature,
        base_temperature=aux_args.supcon_loss_base_temperature if aux_args.supcon_loss_base_temperature is not None else aux_args.supcon_loss_temperature,
        contrast_mode=aux_args.supcon_loss_contrast_mode
    )
    def compute_loss_func(model_outputs, labels, num_items_in_batch = None):
        logits = model_outputs["logits"]
        features = model_outputs.get("features", None) # Some model outputs are missing `features` tensor
        loss = 0.0
        loss_norm_denom = 0.0

        if aux_args.ce_loss_multiplier is not None:
            ce_loss = F.cross_entropy(logits, labels)
            loss += ce_loss * aux_args.ce_loss_multiplier
            loss_norm_denom += aux_args.ce_loss_multiplier if aux_args.loss_reduction == "w_mean" else 1.0

        if aux_args.focal_loss_multiplier is not None:
            focal_loss = F_focal_loss(logits, labels)
            loss += focal_loss * aux_args.focal_loss_multiplier
            loss_norm_denom += aux_args.focal_loss_multiplier if aux_args.loss_reduction == "w_mean" else 1.0

        if aux_args.supcon_loss_multiplier is not None and features is not None:
            supcon_loss = F_supcon_loss(features, labels)
            loss += supcon_loss * aux_args.supcon_loss_multiplier
            loss_norm_denom += aux_args.supcon_loss_multiplier if aux_args.loss_reduction == "w_mean" else 1.0

        if aux_args.loss_reduction != "sum":
            loss /= loss_norm_denom + 1e-12
        return loss

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    # TODO: Modify config params at runtime
    if isinstance(config, ViTHybridConfig):
        if aux_args.hidden_dropout is not None:
            config.hidden_dropout_prob = aux_args.hidden_dropout
        if aux_args.attention_dropout is not None:
            config.attention_probs_dropout_prob = aux_args.attention_dropout

    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # TODO: Make sure that images are not preprocessed multiple times (here and then in model image processor)
    # Define torchvision transforms to be applied to each image.
    if isinstance(image_processor, TimmWrapperImageProcessor):
        _train_transforms = image_processor.train_transforms
        _val_transforms = image_processor.val_transforms
    elif isinstance(image_processor, ViTHybridImageProcessor):
        hybrid_vit_transform = Lambda(
            lambda im: image_processor(images=im, return_tensors="pt")["pixel_values"][0]
        )
        _train_transforms = hybrid_vit_transform
        _val_transforms = hybrid_vit_transform
    else:
        if "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        else:
            size = (image_processor.size["height"], image_processor.size["width"])

        # Create normalization transform
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std"):
            normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        else:
            normalize = Lambda(lambda x: x)
        _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(val_transforms)

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        compute_loss_func=compute_loss_func,
        processing_class=image_processor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()