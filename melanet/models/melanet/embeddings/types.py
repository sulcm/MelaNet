import torch
import numpy as np
import pandas as pd

from typing import Optional, Union
from itertools import repeat
from dataclasses import dataclass, field, fields

from .utils import normalize_embeddings
from ..utils import tensor2numpy


MELANET_FEATURES_PREFIX = "melanet_features"


@dataclass
class FeatureExtractorOutput:
    ft_embeddings: Optional[torch.Tensor] = field(default=None)
    zero_shot_embeddings: Optional[torch.Tensor] = field(default=None)

    @classmethod
    def get_feature_names(cls) -> list[str]:
        return [
            f.name
            for f in fields(cls)
        ]

    def have_ft_embeddings(self) -> bool:
        return self.ft_embeddings is not None

    def have_zero_shot_embeddings(self) -> bool:
        return self.zero_shot_embeddings is not None

    def sum(self, alpha: float, normalize: bool = False, pre_norm: bool = False) -> torch.Tensor:
        if  self.have_ft_embeddings() and self.have_zero_shot_embeddings():
            if pre_norm:
                self.l2_normalize()
            w_sum = alpha * self.ft_embeddings + (1.0 - alpha) * self.zero_shot_embeddings
            return normalize_embeddings(w_sum) if normalize else w_sum
        elif self.have_ft_embeddings():
            return normalize_embeddings(self.ft_embeddings) if normalize else self.ft_embeddings
        elif self.have_zero_shot_embeddings():
            return normalize_embeddings(self.zero_shot_embeddings) if normalize else self.zero_shot_embeddings
        else:
            raise ValueError("Output has no embeddings")

    def concat(self, normalize: bool = False, pre_norm: bool = False) -> torch.Tensor:
        if  self.have_ft_embeddings() and self.have_zero_shot_embeddings():
            if pre_norm:
                self.l2_normalize()
            concat_embeds = torch.cat(
                [self.ft_embeddings, self.zero_shot_embeddings],
                dim=-1
            )
            return normalize_embeddings(concat_embeds) if normalize else concat_embeds
        elif self.have_ft_embeddings():
            return normalize_embeddings(self.ft_embeddings) if normalize else self.ft_embeddings
        elif self.have_zero_shot_embeddings():
            return normalize_embeddings(self.zero_shot_embeddings) if normalize else self.zero_shot_embeddings
        else:
            raise ValueError("Output has no embeddings")

    def l2_normalize(self) -> "FeatureExtractorOutput":
        if self.have_ft_embeddings():
            self.ft_embeddings = normalize_embeddings(self.ft_embeddings)
        if self.have_zero_shot_embeddings():
            self.zero_shot_embeddings = normalize_embeddings(self.zero_shot_embeddings)
        return self

    def to_dict(self, values_as_list: bool = False) -> dict[str, Optional[Union[np.ndarray, list[np.ndarray]]]]:
        _obj_dict = {}
        _obj_dict["ft_embeddings"] = tensor2numpy(self.ft_embeddings) if self.have_ft_embeddings() else None
        if values_as_list and self.have_ft_embeddings():
            _obj_dict["ft_embeddings"] = list(_obj_dict["ft_embeddings"])
        _obj_dict["zero_shot_embeddings"] = tensor2numpy(self.zero_shot_embeddings) if self.have_zero_shot_embeddings() else None
        if values_as_list and self.have_zero_shot_embeddings():
            _obj_dict["zero_shot_embeddings"] = list(_obj_dict["zero_shot_embeddings"])
        return _obj_dict

    def to_list(self) -> list[dict[str, Optional[np.ndarray]]]:
        _obj_dict = self.to_dict()
        return [
            {
                "ft_embeddings": ft_embeds,
                "zero_shot_embeddings": zs_embeds
            }
            for ft_embeds, zs_embeds in zip(
                _obj_dict["ft_embeddings"] if self.have_ft_embeddings() else repeat(None),
                _obj_dict["zero_shot_embeddings"] if self.have_zero_shot_embeddings() else repeat(None)
            )
        ]

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.to_dict(values_as_list=True)
        )