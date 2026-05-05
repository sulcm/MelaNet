import torch
import numpy as np

from typing import Optional, Union, Callable

from .learnable_base import LearnableAdapter
from .adapter_wrapper import AdapterWrapper
from .config import FeatureAdapterConfig
from .types import AdapterOutput
from ..embeddings.config import FeatureExtractorConfig
from ..embeddings.types import FeatureExtractorOutput
from ..zero_shot.config import ZeroShotConfig
from ..melanet_wrapper import MelaNet


class LearnableAdapterWithMelaNetWrapper(LearnableAdapter):
    def __init__(
        self,
        adapter_name_or_path: Optional[str] = None,
        adapter_type: Optional[str] = None,
        adapter_config: Union[FeatureAdapterConfig, str, None] = None,
        model_name: Optional[str] = None,
        feature_extractor_config: Optional[FeatureExtractorConfig] = None,
        zero_shot_model_name: Optional[str] = None,
        zero_shot_config: Optional[ZeroShotConfig] = None,
        prediction_mapping: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        device: str = "cuda"
    ):
        super(LearnableAdapterWithMelaNetWrapper, self).__init__()

        self.melanet_model = MelaNet(
            model_name=model_name,
            is_feature_extractor=True,
            feature_extractor_config=FeatureExtractorConfig.from_cli(feature_extractor_config) if feature_extractor_config is not None else None,
            zero_shot_model_name=zero_shot_model_name,
            zero_shot_config=ZeroShotConfig.from_cli(zero_shot_config) if zero_shot_config is not None else None,
            prediction_mapping=prediction_mapping,
            device=device
        )

        if adapter_name_or_path is not None:
            adapter_model = AdapterWrapper.from_pretrained(adapter_name_or_path)
        elif adapter_type is not None and isinstance(adapter_config, FeatureAdapterConfig):
            adapter_model = AdapterWrapper.from_config(
                adapter_type=adapter_type,
                config=adapter_config
            )
        elif adapter_type is not None and adapter_config is not None:
            _adapter_config = FeatureAdapterConfig.from_args(adapter_config)
            adapter_model = AdapterWrapper.from_config(
                adapter_type=adapter_type,
                config=_adapter_config
            )
        else:
            raise ValueError(
                "You must provide `adapter_model` OR (`adapter_type` and `adapter_config`)"
            )
        self.adapter: LearnableAdapter = adapter_model

    @property
    def adapter_type(self) -> str:
        return self.adapter.adapter_type

    def forward(self, image, **kwargs) -> AdapterOutput:
        with torch.no_grad():
            melanet_output = self.melanet_model(image)

        if self.adapter.adapter_type == "fusion":
            assert isinstance(melanet_output, FeatureExtractorOutput)
            output = self.adapter([
                melanet_output.ft_embeddings,
                melanet_output.zero_shot_embeddings,
            ])
        else:
            if isinstance(melanet_output, FeatureExtractorOutput):
                if melanet_output.have_ft_embeddings():
                    output = self.adapter(melanet_output.ft_embeddings)
                elif melanet_output.have_zero_shot_embeddings():
                    output = self.adapter(melanet_output.ft_embeddings)
                else:
                    raise ValueError("Must have some embeddings")
            else:
                output = self.adapter(melanet_output)

        return output

    def train(self, mode: bool = True):
        super().train(mode)
        # force backbone to stay frozen
        self.melanet_model.eval()
        self.melanet_model.freeze_parameters()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # force device
        self.melanet_model.to(*args, **kwargs)
        return self

    @classmethod
    def from_config(cls, config: FeatureAdapterConfig) -> "LearnableAdapter":
        raise OSError(
            f"{cls.__class__.__name__} is designed to be used only for training purposes."
            f"For adapter initialization use {AdapterWrapper.__class__.__name__} or {MelaNet.__class__.__name__}."
        )

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "LearnableAdapter":
        raise OSError(
            f"{cls.__class__.__name__} is designed to be used only for training purposes."
            f"For adapter initialization use {AdapterWrapper.__class__.__name__} or {MelaNet.__class__.__name__}."
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: FeatureAdapterConfig) -> "LearnableAdapter":
        raise OSError(
            f"{cls.__class__.__name__} is designed to be used only for training purposes."
            f"For adapter initialization use {AdapterWrapper.__class__.__name__} or {MelaNet.__class__.__name__}."
        )

    def save_as_pretrained(self, save_path: str, allow_overwrite: bool = True) -> None:
        self.adapter.save_as_pretrained(
            save_path=save_path,
            allow_overwrite=allow_overwrite
        )