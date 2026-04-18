import torch
import numpy as np

from typing import Optional, Union

from transformers import AutoImageProcessor, AutoModelForImageClassification

from .embeddings.config import FeatureExtractorConfig, create_default_feature_extractor_config
from .embeddings.types import FeatureExtractorOutput
from .embeddings.utils import normalize_embeddings

from .zero_shot.config import ZeroShotConfig, create_default_zero_shot_config
from .zero_shot.models.hf_wrapper import HfModelWrapper
from .zero_shot.zero_shot_wrapper import ZeroShotModel

from .adapters.adapter_wrapper import AdapterWrapper

from .utils import tensor2numpy, resolve_device


class MelaNet():
    def __init__(
        self,
        model_name: Optional[str] = None,
        is_feature_extractor: bool = False,
        feature_extractor_config: Optional[FeatureExtractorConfig] = None,
        zero_shot_model_name: Optional[str] = None,
        zero_shot_config: Optional[ZeroShotConfig] = None,
        ft_model_adapter_name: Optional[str] = None,
        zero_shot_model_adapter_name: Optional[str] = None,
        models_fusion_adapter_name: Optional[str] = None,
        device: str = "cuda"
    ):
        self.device = resolve_device(device)
        self.is_feature_extractor = is_feature_extractor

        if self.is_feature_extractor:
            assert model_name is not None or zero_shot_model_name is not None, "At least one of `model_name` or `zero_shot_model_name` must be provided"
            self.__feature_extractor_config = feature_extractor_config if feature_extractor_config is not None else create_default_feature_extractor_config()
            if model_name is not None:
                self.ft_model = HfModelWrapper(model_name=model_name, config=zero_shot_config, device=self.device)

                self.ft_model_adapter = AdapterWrapper.from_pretrained(ft_model_adapter_name) if ft_model_adapter_name is not None else None
                if self.ft_model_adapter is not None:
                    self.ft_model_adapter.eval().to(self.device)
            else:
                self.ft_model = None
                self.ft_model_adapter = None

            if zero_shot_model_name is not None:
                zero_shot_config = zero_shot_config if zero_shot_config is not None else create_default_zero_shot_config()
                self.zero_shot_model = ZeroShotModel(model_name=zero_shot_model_name, config=zero_shot_config, device=self.device)

                self.zero_shot_model_adapter = AdapterWrapper.from_pretrained(zero_shot_model_adapter_name) if zero_shot_model_adapter_name is not None else None
                if self.zero_shot_model_adapter is not None:
                    self.zero_shot_model_adapter.eval().to(self.device)
            else:
                self.zero_shot_model = None
                self.zero_shot_model_adapter = None

            if model_name is not None and zero_shot_model_name is not None:
                self.models_fusion_adapter = AdapterWrapper.from_pretrained(models_fusion_adapter_name) if models_fusion_adapter_name is not None else None
                if self.models_fusion_adapter is not None:
                    self.models_fusion_adapter.eval().to(self.device)
            else:
                self.models_fusion_adapter = None
        else:
            assert model_name is not None, "For classification must specify fine-tuned model name using `model_name` parameter"
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.eval().to(self.device)

    def get_labels(self) -> Optional[list[str]]:
        if self.is_feature_extractor:
            return None
        else:
            # Each model initialized from `AutoModelForImageClassification` has config `PreTrainedConfig` with `label2id` attribute
            return list(
                self.model.config.label2id.keys()
            )

    @property
    def feature_extractor_config(self) -> Optional[FeatureExtractorConfig]:
        if self.is_feature_extractor:
            return self.__feature_extractor_config
        else:
            return None

    @property
    def zero_shot_config(self) -> Optional[ZeroShotConfig]:
        if self.is_feature_extractor:
            if self.zero_shot_model is not None:
                return self.zero_shot_model.config
            else:
                return None
        else:
            return None

    def forward(self, image, return_logits: bool = False, **kwargs) -> np.ndarray:
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        logits = outputs.logits
        if return_logits:
            return tensor2numpy(logits)
        else:
            predicted_class_idx = tensor2numpy(logits.argmax(-1))
            return predicted_class_idx

    def extract_features(self, image, text = None, **kwargs) -> Union[torch.Tensor, FeatureExtractorOutput]:
        if self.ft_model is not None:
            ft_embeds = self.ft_model.extract_features(image=image)
        else:
            ft_embeds = None

        if self.zero_shot_model is not None:
            zero_shot_embeds = self.zero_shot_model.extract_features(image=image, text=text)
        else:
            zero_shot_embeds = None

        output = FeatureExtractorOutput(
            ft_embeddings=ft_embeds,
            zero_shot_embeddings=zero_shot_embeds
        )

        if output.have_ft_embeddings() and self.ft_model_adapter is not None:
            ft_model_adapter_output = self.ft_model_adapter.forward(output.ft_embeddings)
            output.ft_embeddings = ft_model_adapter_output.adapter_output
        if output.have_zero_shot_embeddings() and self.zero_shot_model_adapter is not None:
            zero_shot_model_adapter_output = self.zero_shot_model_adapter.forward(output.zero_shot_embeddings)
            output.zero_shot_embeddings = zero_shot_model_adapter_output.adapter_output
        if (
            (output.have_ft_embeddings() and output.have_zero_shot_embeddings())
            and
            self.models_fusion_adapter is not None
        ):
            models_fusion_adapter_output = self.models_fusion_adapter.forward([
                output.ft_embeddings,
                output.zero_shot_embeddings
            ])
            fused_output = models_fusion_adapter_output.adapter_output
            return normalize_embeddings(fused_output) if self.__feature_extractor_config.normalize_output else fused_output

        if self.__feature_extractor_config.output_type == "object":
            return output.l2_normalize() if self.__feature_extractor_config.normalize_output else output
        elif self.__feature_extractor_config.output_type == "sum":
            return output.sum(
                alpha=self.__feature_extractor_config.alpha,
                normalize=self.__feature_extractor_config.normalize_output,
                pre_norm=self.__feature_extractor_config.pre_norm
            )
        elif self.__feature_extractor_config.output_type == "concat":
            return output.concat(
                normalize=self.__feature_extractor_config.normalize_output,
                pre_norm=self.__feature_extractor_config.pre_norm
            )
        else:
            raise ValueError(
                "Unsupported `output_type` for zero-shot feature extraction. Must be one of: 'object', 'sum', 'concat'"
            )

    def __call__(self, image, *, text = None, return_logits: bool = False, **kwargs):
        if self.is_feature_extractor:
            return self.extract_features(
                image=image,
                text=text
            )
        else:
            return self.forward(
                image=image,
                return_logits=return_logits
            )