import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union


class SeesawLoss(nn.Module):
    """
    Implementation of Seesaw Loss (Cao et al., CVPR 2021)
    https://arxiv.org/abs/2008.10032

    Differences from standard CE:
        - Mitigation factor downweights head -> tail confusion
        - Compensation factor rebalances tail -> head confusion
        - Global cumulative class frequencies are maintained
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.8,
        q: float = 2.0,
        eps: float = 1e-12,
        mask_mode: bool = True,
        keep_target_logits: bool = True,
        label_weights: Optional[list[float]] = None,
        reduction: str = "mean",
        device: Union[torch.device, str, None] = None
    ) -> None:
        """Initialize Seesaw Loss

        Args:
            num_classes (int): Number of classes.
            p (float, optional): Power for mitigation factor. Defaults to 0.8.
            q (float, optional): Power for compensation factor. Defaults to 2.0.
            eps (float, optional): Small epsilon for numerical stability. Defaults to 1e-12.
            mask_mode (bool, optional): If True, apply masking logic from the paper
                                        (only apply mitigation when ratio<1, etc.).
                                        Defaults to True.
            keep_target_logits (bool, optional): If True, only non-target logits are adjusted.
                                                 Defaults to True.
            label_weights (Optional[list[float]], optional): Label-wise loss weight. Defaults to None.
            reduction (str, optional): Output loss reduction.
                                       Posible values: "none" | "mean" | "sum";
                                       Defaults to "mean".
            device: (torch.device | str | None): Device where to store buffer. Must be the same as expected logits.
                                                 Defaults to None.
        """
        super(SeesawLoss, self).__init__()

        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps
        self.mask_mode = mask_mode
        self.keep_target_logits = keep_target_logits
        self.reduction = reduction

        if label_weights is not None and isinstance(label_weights, (list, tuple)):
            assert len(label_weights) == num_classes, "Size on `label_weights` do not match the number of classes"
            self.label_weights = torch.tensor(label_weights)
        else:
            self.label_weights = None

        # Global cumulative sample count (persisting across batches)
        self.register_buffer("cum_samples", torch.zeros(num_classes, dtype=torch.float, device=device))

    @torch.no_grad()
    def update_class_counts(self, targets: torch.Tensor):
        """Accumulate class counts globally across training."""
        unique, counts = torch.unique(targets, return_counts=True)
        self.cum_samples[unique.to(self.cum_samples.device)] += counts.to(self.cum_samples.device).float()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (Tensor): [N, C] raw model outputs.
            targets (Tensor): [N] ground truth labels.

        Returns:
            Tensor: Scalar loss.
        """
        assert logits.dim() == 2 and logits.size(1) == self.num_classes, \
            "logits must have shape [N, num_classes]"

        device = logits.device
        N, C = logits.size()
        seesaw_weights = logits.new_ones(logits.size()) # [N, C]

        # Update cumulative sample counts
        self.update_class_counts(targets)

        # --- 1. Compute mitigation factor ---
        if self.p > 0:
            freq = self.cum_samples.clamp(min=1.0)
            ratio_matrix = freq.unsqueeze(0) / freq.unsqueeze(1)  # [C, C]

            if self.mask_mode:
                # Apply mitigation only when ratio < 1 (tail vs head)
                index = (ratio_matrix < 1.0).float()
                sample_weights = torch.pow(ratio_matrix, self.p) * index + (1 - index)
            else:
                # Always apply mitigation smoothly
                sample_weights = torch.pow(torch.clamp(ratio_matrix, max=1.0), self.p)

            mitigation_factor = sample_weights[targets.long(), :]  # [N, C]
            seesaw_weights *= mitigation_factor

        # --- 2. Compute compensation factor ---
        if self.q > 0:
            probs = F.softmax(logits.detach(), dim=1)
            self_scores = probs[torch.arange(N, device=device), targets.long()]  # [N]
            score_matrix = probs / self_scores[:, None].clamp(min=self.eps)

            if self.mask_mode:
                # Only apply compensation when prob_j > prob_i
                index = (score_matrix > 1.0).float()
                compensation_factor = torch.pow(score_matrix, self.q) * index + (1 - index)
            else:
                compensation_factor = torch.pow(torch.clamp(score_matrix, min=1.0), self.q)

            seesaw_weights *= compensation_factor

        # --- 3. Adjust logits ---
        if self.keep_target_logits:
            # Only adjust non-target class logits
            one_hot = F.one_hot(targets, num_classes=C).float()
            adjusted_logits = logits + (seesaw_weights.log() * (1.0 - one_hot))
        else:
            adjusted_logits = logits + seesaw_weights.log()

        # --- 4. Compute standard cross-entropy on adjusted logits ---
        loss = F.cross_entropy(
            adjusted_logits,
            targets,
            weight=self.label_weights,
            reduction=self.reduction,
        )
        return loss