import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class InfoNCE(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean",
        negative_mode: str = "unpaired"
    ):
        super(InfoNCE, self).__init__()

        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        negative_targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must be 2D tensors")
        if targets is not None:
            if targets.ndim != 2:
                raise ValueError("targets must be 2D tensors")
            if features.shape != targets.shape:
                raise ValueError("features and targets must have the same shape")
        if negative_targets is not None:
            if self.negative_mode == "unpaired" and negative_targets.ndim != 2:
                raise ValueError("unpaired negative_targets must be 2D")
            if self.negative_mode == "paired" and negative_targets.ndim != 3:
                raise ValueError("paired negative_targets must be 3D")

        features = F.normalize(features, p=2, dim=-1)
        if targets is not None:
            targets = F.normalize(targets, p=2, dim=-1)
        if negative_targets is not None:
            negative_targets = F.normalize(negative_targets, p=2, dim=-1)

        if negative_targets is not None:
            # Positive logits: (N, 1)
            pos_logits = torch.sum(
                features * targets if targets is not None else torch.pow(features, 2),
                dim=1,
                keepdim=True
            )

            if self.negative_mode == "unpaired":
                neg_logits = features @ negative_targets.T  # (N, M)
            else:
                neg_logits = torch.einsum("nd,nmd->nm", features, negative_targets)   # (N, M)

            logits = torch.cat([pos_logits, neg_logits], dim=1)
            labels = torch.zeros_like(features, dtype=torch.long)
        else:
            if targets is not None:
                logits = features @ targets.T
                labels = torch.arange(features.size(0), device=features.device)
            else:
                # Mask self-similarity
                repeated_features = torch.cat([features, features], dim=0)
                logits = repeated_features @ repeated_features.T
                mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
                logits = logits.masked_fill(mask, float("-inf"))
                labels = torch.arange(features.size(0), device=features.device)
                labels = torch.cat([labels + features.size(0), labels], dim=0)

        loss = F.cross_entropy(
            torch.div(logits, self.temperature),
            labels,
            reduction=self.reduction
        )
        return loss