import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers

from typing import cast, Optional, Callable, Literal, Any
from dataclasses import dataclass, field
from functools import partial
from collections import Counter

from datasets import load_dataset, load_from_disk

from transformers import (
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from torchmetrics.functional import (
    auroc,
    average_precision,
    accuracy,
    specificity,
    recall,
    f1_score,
    precision,
    negative_predictive_value,
)

# TODO: Workaround when launching this script from different locations
if __name__ == "__main__":
    # Script is run directly
    from loss_functions import FocalLoss, SupConLoss, SeesawLoss
    from metacentrum_utils import load_dataset_from_scratch, METACENTRUM_SCRATCH_PREFIX
    from train_utils import generate_id, check_report_to_integration
    from melanet.adapters import (
        ADAPTER_TYPES,
        AdapterWrapper,
        LearnableAdapter,
        FeatureAdapterConfig
    )
else:
    # Script is being imported or used from different location
    from .loss_functions import FocalLoss, SupConLoss, SeesawLoss
    from .metacentrum_utils import load_dataset_from_scratch, METACENTRUM_SCRATCH_PREFIX
    from .train_utils import generate_id, check_report_to_integration
    from .melanet.adapters import (
        ADAPTER_TYPES,
        AdapterWrapper,
        LearnableAdapter,
        FeatureAdapterConfig
    )


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: str = field(
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
        default=None, metadata={"help": "Percent to split off of train for validation."}
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
    feature_column_names: str = field(
        default="features",
        metadata={"help": "The name of the dataset columns containing the extracted features in format \"features_A,features_B,...\". Defaults to 'features'."},
    )
    label_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
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

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class AdapterArguments:
    """
    Arguments pertaining to which model adapter we are going to train.
    """

    adapter_output_path: str = field(
        metadata={"help": "Path where to save trained adapter."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier in format `<adapter_type>---<adapter_path>`."},
    )
    adapter_type: Optional[str] = field(
        default=None,
        metadata={"help": f"If training from scratch, pass a model type from the list: {ADAPTER_TYPES}"},
    )
    adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Config from which adapter will be initialized."}
    )


@dataclass
class AuxiliaryArguments:
    """
    Auxiliary arguments to specify/define for model/data/training
    """

    # Training
    task: Literal["binary", "multiclass", "multilabel", "features"] = field(
        metadata={"help": "Specify classification task / objective. Possible values are 'binary', 'multiclass', or 'multilabel'."}
    )
    additional_optim_kwargs: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                'Optimazer specific arguments in JSON structure, like: `"{\"momentum\":0.9,\"nesterov\":true}"`.'
                ' Usefull when running sweeps. Escape double quotes using `\\`. Will be applied only if `TrainingArguments.optim_args` is None.'
            )
        }
    )
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
    ## Seesaw
    seesaw_loss_multiplier: Optional[float] = field(
        default=None,
        metadata={"help": "Value that multiplies Seesaw loss. If `None` then do not apply Seesaw loss."}
    )
    seesaw_p: float = field(
        default=0.8,
        metadata={"help": "Power for mitigation factor."}
    )
    seesaw_q: float = field(
        default=2.0,
        metadata={"help": "Power for compensation factor."}
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


def _get_metrics(
    task: Literal["binary", "multiclass", "multilabel", "features"],
    num_classes: Optional[int] = None,
    average: Optional[str] = "macro",
):
    if task not in ("binary", "features"):
        assert num_classes is not None, "For multiclass tasks provide number of classes"

    metrics: dict[str, Callable] = {
        "roc_auc": partial(
            auroc,
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "ap": partial(
            average_precision,
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "accuracy": partial(
            accuracy,
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "specificity": partial(
            specificity,
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "sensitivity": partial( # or recall
            recall,
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "f1": partial(
            f1_score,
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "ppv": partial( # or precision
            precision,
            task=task,
            num_classes=num_classes,
            average=average
        ),
        "npv": partial(
            negative_predictive_value,
            task=task,
            num_classes=num_classes,
            average=average
        ),
    }

    if task == "features":
        for unsupported_metric in ("ap", "roc_auc"):
            _ = metrics.pop(unsupported_metric, None)

    return metrics


def main(args: Optional[dict[str, Any]] = None):
    parser = HfArgumentParser((AdapterArguments, DataTrainingArguments, TrainingArguments, AuxiliaryArguments))
    if args is not None:
        parsed_args = parser.parse_dict(args=args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If passed only one argument to the script then assume it's the path to a json file
        parsed_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        parsed_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args, aux_args = cast(tuple[AdapterArguments, DataTrainingArguments, TrainingArguments, AuxiliaryArguments], parsed_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, set the log level at info here to have that as a default
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Process log summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Adapter model parameters {model_args}")

    # Set seed (before initializing model in case there are no pretrained weights)
    set_seed(training_args.seed)

    # Initialize dataset and prepare it for the 'image-classification' task
    if os.path.exists(data_args.dataset_name):
        # Load from local path
        logger.info(f"Loading dataset {data_args.dataset_name} from local path")
        dataset = load_from_disk(
            dataset_path=data_args.dataset_name
        )
    elif data_args.dataset_name.startswith(METACENTRUM_SCRATCH_PREFIX):
        # Load from scratch directory on Metacentrum
        logger.info(f"Loading dataset {data_args.dataset_name} from scratch storage")
        dataset = load_dataset_from_scratch(
            data_args.dataset_name
        )
    else:
        # Pull from HF or load it from cache
        logger.info(f"Loading dataset {data_args.dataset_name} from HF Hub")
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=data_args.cache_dir,
            token=data_args.token,
            trust_remote_code=data_args.trust_remote_code,
        )
    logger.info(f"Loaded dataset: {dataset}")

    # Validate feature and label columns
    feature_columns = [col.strip() for col in data_args.feature_column_names.split(",")]
    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    if any(f_col not in dataset_column_names for f_col in feature_columns):
        raise ValueError(
            f"--feature_column_names {data_args.feature_column_names} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--feature_column_names` to the correct feature columns - some of "
            f"{', '.join(dataset_column_names)}."
        )
    if data_args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct label column - one of "
            f"{', '.join(dataset_column_names)}."
        )

    # Prepare label mappings
    labels = dataset["train"].features[data_args.label_column_name].names
    labels_int = []
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
        labels_int.append(i)
    logger.info(f"Loaded dataset has {len(labels)} ground truth labels")

    metrics = _get_metrics(
        task=aux_args.task,
        num_classes=len(labels_int),
        average="macro"
    )
    logger.info(f"Using metrics {metrics}")

    # Definition of `compute_metrics` function
    # Input expects `EvalPrediction` object (a namedtuple with a `predictions` and `label_ids` field)
    # Output has to be a dictionary string to float
    def compute_metrics(p):
        """Computes metrics on a batch of predictions"""
        logits = torch.tensor(p.predictions)
        target = torch.tensor(p.label_ids)
        probs = F.softmax(logits, dim=-1)

        results = {
            metric_name: metric(
                preds=probs,
                target=target
            ).float()
            for metric_name, metric in metrics.items()
        }
        return results

    # Custom loss calculation
    losses: list[tuple[float, nn.Module]] = []
    if (
        aux_args.ce_loss_multiplier is None
        and aux_args.focal_loss_multiplier is None
        and aux_args.supcon_loss_multiplier is None
        and aux_args.seesaw_loss_multiplier is None
    ):
        logger.warning(
            "Missing specified loss function. Falling back to CE loss with multiplier set to 1.0"
        )
        aux_args.ce_loss_multiplier = 1.0

    if aux_args.ce_loss_multiplier is not None:
        F_ce_loss = nn.CrossEntropyLoss()
        losses.append((
            aux_args.ce_loss_multiplier,
            F_ce_loss
        ))
    if aux_args.focal_loss_multiplier is not None:
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
        losses.append((
            aux_args.focal_loss_multiplier,
            F_focal_loss
        ))
    if aux_args.seesaw_loss_multiplier is not None:
        F_seesaw_loss = SeesawLoss(
            num_classes=len(labels),
            p=aux_args.seesaw_p,
            q=aux_args.seesaw_q,
            device=training_args.device
        )
        losses.append((
            aux_args.seesaw_loss_multiplier,
            F_seesaw_loss
        ))
    if aux_args.supcon_loss_multiplier is not None:
        F_supcon_loss = SupConLoss(
            temperature=aux_args.supcon_loss_temperature,
            base_temperature=aux_args.supcon_loss_base_temperature if aux_args.supcon_loss_base_temperature is not None else aux_args.supcon_loss_temperature,
            contrast_mode=aux_args.supcon_loss_contrast_mode
        )
        losses.append((
            aux_args.supcon_loss_multiplier,
            F_supcon_loss
        ))
    logger.info(f"Using loss {losses}")

    # Create override method for `compute_loss_func`
    # Input expects model_outputs (`dict` or `AdapterOutput`), labels (`Tensor`), and num_items_in_batch (`Tensor`, optional)
    # Output should be scalar loss (`float`)
    def compute_loss_func(model_outputs, labels, num_items_in_batch = None):
        adapter_output = model_outputs["adapter_output"]
        eps = 1e-12
        loss = 0.0
        loss_norm_denom = 0.0

        for loss_multiplier, F_loss in losses:
            if isinstance(F_loss, SupConLoss):
                _loss = F_loss(adapter_output, labels)
            else:
                _loss = F_loss(adapter_output, labels)
            loss += _loss * loss_multiplier
            loss_norm_denom += (loss_multiplier if aux_args.loss_reduction == "w_mean" else 1.0)

        if aux_args.loss_reduction != "sum":
            loss /= (loss_norm_denom + eps)
        return loss

    if model_args.adapter_name_or_path is not None:
        logger.info(f"Initializing adapter from path {model_args.adapter_name_or_path}")
        adapter_model = AdapterWrapper.from_pretrained(model_args.adapter_name_or_path)
    elif model_args.adapter_type is not None and model_args.adapter_config is not None:
        try:
            logger.info("Initializing adapter from `adapter_config`. Trying to load if as JSON ...")
            _adapter_config_kwargs = json.loads(model_args.adapter_config)
            adapter_config = FeatureAdapterConfig(**_adapter_config_kwargs)
        except Exception:
            logger.info("Loading `adapter_config` as JSON failed falling back to CLI style kwargs")
            adapter_config = FeatureAdapterConfig.from_cli(model_args.adapter_config)
        adapter_model = AdapterWrapper.from_config(
            adapter_type=model_args.adapter_type,
            config=adapter_config
        )
    else:
        raise ValueError(
            "You must provide `adapter_model` OR (`adapter_type` and `adapter_config`)"
        )
    logger.info(f"Initialized adapter model {adapter_model}")

    if aux_args.additional_optim_kwargs:
        try:
            additional_optim_kwargs = json.loads(aux_args.additional_optim_kwargs)
        except Exception:
            logger.warning(
                f"`additional_optim_kwargs` could not be parsed, try to fix JSON formatting of\n{aux_args.additional_optim_kwargs}"
            )
            additional_optim_kwargs = None
    else:
        additional_optim_kwargs = None
    additional_optim_kwargs = additional_optim_kwargs or {}

    if isinstance(adapter_model, LearnableAdapter):
        optim_cls, base_optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args, adapter_model)
        optim_kwargs = {
            **base_optim_kwargs,
            **additional_optim_kwargs
        }
        optimizer_cls_and_kwargs = (optim_cls, optim_kwargs)
    else:
        optimizer_cls_and_kwargs = None
    logger.info(f"Using optimizer {optimizer_cls_and_kwargs}")

    if isinstance(adapter_model, LearnableAdapter) and check_report_to_integration("tensorboard", training_args.report_to):
        if training_args.logging_dir is not None:
            training_args.logging_dir = os.path.join(training_args.logging_dir, f"run_{generate_id(as_datetime=True)}_{generate_id()}")
        logger.info(
            f"Logging to TensorBoard: {training_args.logging_dir if training_args.logging_dir is not None else "Default HF location"}"
        )

    # Training
    logger.info(f"Starting training of `{adapter_model.adapter_type}` adapter")
    if adapter_model.adapter_type == "pca":
        ft_col = feature_columns[0]
        X_train = np.asarray(dataset["train"][ft_col])
        adapter_model.fit(X_train)
    elif isinstance(adapter_model, LearnableAdapter):
        if adapter_model.adapter_type == "fusion":
            ft_col = feature_columns

            def train_transforms(example_batch):
                """Apply _train_transforms across a batch."""
                for ft_c in ft_col:
                    example_batch[ft_c] = [
                        torch.tensor(ft) for ft in example_batch[ft_c]
                    ]
                return example_batch

            def val_transforms(example_batch):
                """Apply _val_transforms across a batch."""
                for ft_c in ft_col:
                    example_batch[ft_c] = [
                        torch.tensor(ft) for ft in example_batch[ft_c]
                    ]
                return example_batch
        else:
            ft_col = feature_columns[0]

            def train_transforms(example_batch):
                """Apply _train_transforms across a batch."""
                example_batch[ft_col] = [
                    torch.tensor(ft) for ft in example_batch[ft_col]
                ]
                return example_batch

            def val_transforms(example_batch):
                """Apply _val_transforms across a batch."""
                example_batch[ft_col] = [
                    torch.tensor(ft) for ft in example_batch[ft_col]
                ]
                return example_batch

        dataset["train"].set_transform(train_transforms)
        dataset["validation"].set_transform(val_transforms)

        adapter_model.fit(
            training_args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            feature_column_names=ft_col,
            label_column_name=data_args.label_column_name,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics
        )
    else:
        raise ValueError(
            f"Trying to train unsupported adapter model. Must be one of {ADAPTER_TYPES}."
        )

    logger.info(f"Saving adapter model to {model_args.adapter_output_path}")
    adapter_model.save_as_pretrained(
        save_path=model_args.adapter_output_path,
        allow_overwrite=training_args.overwrite_output_dir
    )


if __name__ == "__main__":
    main()