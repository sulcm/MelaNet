import timm
import torch

from typing import Optional

from timm.data import resolve_data_config, create_transform

from ..config import ZeroShotConfig, create_default_zero_shot_config
from ...embeddings.utils import normalize_embeddings
from ...utils import resolve_device


class TimmModelWrapper():
    def __init__(self, model_name: str, config: Optional[ZeroShotConfig] = None, device: str = "cuda"):
        self.device = resolve_device(device)

        self.config = config if config is not None else create_default_zero_shot_config()

        self.model = timm.create_model(
            model_name,
            num_classes=0,
            pretrained=True,
            global_pool=self.config.global_pool,
            pretrained_strict=False
        )
        self.model = self.model.eval().to(self.device)

        model_config = resolve_data_config(
            pretrained_cfg=self.model.pretrained_cfg if hasattr(self.model, "pretrained_cfg") else self.model.default_cfg,
            args=None,
            model=None,
            use_test_size=True
        )
        self.image_processor = create_transform(
            **model_config,
            is_training=False
        )

        self._not_supports_tensor_input = any(
            transform.__class__.__name__ == "ToTensor" for transform in self.image_processor.transforms
        )

    def extract_features(self, image, **kwargs):
        if not isinstance(image, list):
            image = [image,]
        if self._not_supports_tensor_input:
            image_tensor_proc = torch.stack(
                [self.image_processor(im) for im in image]
            )
        else:
            image_tensor_proc = self.image_processor(image)

        image_features = self.model(
            image_tensor_proc.to(self.device)
        )

        return normalize_embeddings(
            image_features
        ) if self.config.normalize_output else image_features