from typing import Optional

from .config import ZeroShotConfig, create_default_zero_shot_config
from .models import HfModelWrapper, OpenCLIPWrapper, TimmModelWrapper

from ..utils import resolve_device


class ZeroShotModel():
    def __init__(self, model_name: str, config: Optional[ZeroShotConfig] = None, device: str = "cuda"):
        self.device = resolve_device(device)
        self.config = config if config is not None else create_default_zero_shot_config()
        self.architecture = model_name

        if self.config.model_backend == "open_clip":
            self.zero_shot_wrapper = OpenCLIPWrapper(
                model_name=self.architecture,
                config=self.config,
                device=self.device
            )
        elif self.config.model_backend == "timm":
            self.zero_shot_wrapper = TimmModelWrapper(
                model_name=self.architecture,
                config=self.config,
                device=self.device
            )
        else:
            self.zero_shot_wrapper = HfModelWrapper(
                model_name=self.architecture,
                config=self.config,
                device=self.device
            )

    def extract_features(self, image, **kwargs):
        return self.zero_shot_wrapper.extract_features(
            image=image,
            **kwargs
        )

    def __call__(self, image, **kwargs):
        return self.extract_features(
            image=image,
            **kwargs
        )