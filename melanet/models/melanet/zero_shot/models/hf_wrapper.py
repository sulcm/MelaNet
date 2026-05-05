from typing import Optional

from transformers import AutoImageProcessor, AutoModel

from ..config import ZeroShotConfig, create_default_zero_shot_config
from ...embeddings.utils import normalize_embeddings
from ...utils import resolve_device


class HfModelWrapper():
    def __init__(self, model_name: str, config: Optional[ZeroShotConfig] = None, device: str = "cuda"):
        self.device = resolve_device(device)

        self.config = config if config is not None else create_default_zero_shot_config(model_backend="hf")

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)

    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def eval(self):
        self.model.eval()
        return self

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def extract_features(self, image, **kwargs):
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        if self.config.features_output_name is not None:
            if not isinstance(outputs, dict):
                image_features = getattr(outputs, self.config.features_output_name)
            else:
                image_features = outputs.get(self.config.features_output_name)
            assert image_features is not None, f"There is no value in model output named {self.config.features_output_name}"
        else:
            image_features = outputs

        return normalize_embeddings(
            image_features
        ) if self.config.normalize_output else image_features