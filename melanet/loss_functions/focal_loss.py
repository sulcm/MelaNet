"""Based on PyTorch implementation from [public git repository](https://github.com/itakurah/Focal-loss-PyTorch/tree/main)"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union


class FocalLoss(nn.Module):
    def __init__(self, gamma: float=2.0, alpha: Union[float, list[float], None]=None, reduction: str="mean", task_type: str="binary", eps: float=1e-12):
        """
        Unified Focal Loss class for binary, multiclass, and multilabel classification tasks.

        Args:
            gamma (float, optional): Focusing parameter, controls the strength of the modulating factor `(1 - p_t)^gamma`. Defaults to `2.0`.
            alpha (Union[float, list[float], None], optional): Balancing factor, can be a scalar or a tensor for class-wise weights. If `None`, no class balancing is used. Defaults to `None`.
            reduction (str, optional): Specifies the reduction method: "none" | "mean" | "sum". Defaults to "mean".
            task_type (str, optional): Specifies the type of task: "binary", "multiclass", or "multilabel". Defaults to "binary".
            eps (float, optional): Epsilon for numerical stability. Defaults to `1e-12`.
        """
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.eps = eps

        # Handle alpha
        if alpha is not None and isinstance(alpha, (float, list, torch.Tensor)):
            if isinstance(alpha, (float, list)):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the Focal Loss based on the specified task type.

        Args:
            inputs (Tensor): Predictions (logits) from the model.
                Shape:
                    - binary: (batch_size, num_classes)
                    - multiclass: (batch_size, num_classes)
                    - multilabel: (batch_size, num_classes)
            targets (Tensor): Ground truth labels.
                Shape:
                    - binary: (batch_size,)
                    - multiclass: (batch_size,)/(batch_size, num_classes)
                    - multilabel: (batch_size, num_classes)

        Raises:
            ValueError: Invalid task_type
            ValueError: Invalid reduction

        Returns:
            Tensor: Computed Focal loss
        """
        if self.task_type == "binary":
            loss = self.binary_focal_loss(inputs, targets)
        elif self.task_type == "multiclass":
            loss = self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == "multilabel":
            loss = self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multiclass', or 'multilabel'."
            )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(
                f"Unsupported reduction '{self.reduction}'. Use 'none', 'mean', or 'sum'."
            )

    def binary_focal_loss(self, inputs, targets):
        """Focal loss for binary classification."""
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # Compute focal weight
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        loss = (1.0 - p_t) ** self.gamma * bce_loss
        # Apply alpha if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """Focal loss for multiclass classification."""
        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=-1)
        # One-hot encode the targets
        if targets.dim() == 2 and targets.size(1) == inputs.size(-1):
            # Use soft-label variant (for example used when applying MixUp/CutMix augmentations)
            targets_one_hot = targets.float()
        else:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs + self.eps)
        # Compute focal weight
        p_t = (probs * targets_one_hot).sum(dim=-1) # p_t for each sample

        # Compute focal loss
        loss = (1.0 - p_t) ** self.gamma * ce_loss.sum(dim=-1)
        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_factor = (alpha * targets_one_hot).sum(dim=-1)
            loss = alpha_factor * loss
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """Focal loss for multilabel classification."""
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # Compute focal weight
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        loss = (1.0 - p_t) ** self.gamma * bce_loss
        # Apply alpha if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss