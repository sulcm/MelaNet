import open_clip
import torch

from typing import Optional

from ..config import ZeroShotConfig, create_default_zero_shot_config
from ...embeddings.utils import normalize_embeddings
from ...image_utils import make_flat_list_of_images, ensure_pil_image
from ...utils import resolve_device


class OpenCLIPWrapper():
    def __init__(self, model_name: str, config: Optional[ZeroShotConfig] = None, device: str = "cuda"):
        self.device = resolve_device(device)

        self.config = config if config is not None else create_default_zero_shot_config(model_backend="open_clip")

        self.model, _, self.image_processor = open_clip.create_model_and_transforms(model_name)
        self.model.eval().to(self.device)

        self.tokenizer = open_clip.get_tokenizer(model_name) if self.config.add_text_embeddings else None

        self._not_supports_tensor_input = any(
            transform.__class__.__name__ == "ToTensor" for transform in self.image_processor.transforms
        )

    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def eval(self):
        self.model.eval()
        return self

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def extract_features(self, image, text=None, **kwargs):
        if self._not_supports_tensor_input and isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if isinstance(image, torch.Tensor):
            image_tensor_proc = self.image_processor(image)
            image_tensor_proc = image_tensor_proc.unsqueeze(0) if image_tensor_proc.ndim == 3 else image_tensor_proc
        else:
            image = make_flat_list_of_images(image)
            image_tensor_proc = torch.stack([
                self.image_processor(
                    ensure_pil_image(im)
                ) for im in image
            ])

        image_features = self.model.encode_image(
            image_tensor_proc.to(self.device)
        )
        if self.config.pre_norm:
            image_features = normalize_embeddings(image_features)

        if self.config.add_text_embeddings and text is not None:
            text_tensor_proc = self.tokenizer(
                [text,] if isinstance(text, str) else text
            )
            text_features = self.model.encode_text(
                text_tensor_proc.to(self.device)
            )
            if self.config.pre_norm:
                text_features = normalize_embeddings(text_features)
        else:
            text_features = None

        if text_features is None:
            return normalize_embeddings(
                image_features
            ) if self.config.normalize_output else image_features
        else:
            if self.config.output_type == "sum":
                extracted_features = self.config.alpha * image_features + (1.0 - self.config.alpha) * text_features
                return normalize_embeddings(
                    extracted_features
                ) if self.config.normalize_output else extracted_features
            elif self.config.output_type == "concat":
                extracted_features = torch.cat([image_features, text_features], dim=-1)
                return normalize_embeddings(
                    extracted_features
                ) if self.config.normalize_output else extracted_features
            else:
                raise ValueError(
                    "Unsupported `output_type` for zero-shot feature extraction. Must be one of: 'sum', 'concat'"
                )