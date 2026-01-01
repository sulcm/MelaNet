import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss as in:
    https://arxiv.org/pdf/2004.11362.pdf

    Args:
        temperature (float): scaling factor for similarity scores
        base_temperature (float): scaling for loss normalization
        contrast_mode (str): 'all' or 'one'. Defaults to 'all'
            - 'all' -> contrast against all positives
            - 'one' -> contrast only against one positive
    """
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode = "all",
        eps: float = 1e-12
    ):
        super(SupConLoss, self).__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: hidden vector of shape [batch_size, n_views, dim]
            labels: ground truth labels of shape [batch_size] or soft labels of shape [batch_size, num_classes]
            mask: optional contrastive mask of shape [batch_size, batch_size] (can be used instead of labels)
                    mask[i][j] = 1 if sample j has the same class as sample i

        Returns:
            loss value (scalar)
        """
        device = features.device
        if len(features.shape) < 3:
            raise ValueError(f"`features` needs to be [batch_size, n_views, ...], got {features.shape}")
        batch_size, n_views, feature_dim = features.shape

        # Apply L2 normalization (Contrastive losses rely on similarity scores)
        features = F.normalize(features, p=2, dim=2)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # Flatten features: [batch_size * n_views, dim]

        # Handle mask and labels
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`.")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            if labels.dim() == 1:
                # Hard labels [batch_size,]
                labels = labels.contiguous().view(-1, 1)  # [batch_size, 1]
                if labels.shape[0] != batch_size:
                    raise ValueError("Labels shape mismatch with features.")
                mask = torch.eq(labels, labels.T).float().to(device) # binary mask
            elif labels.dim() == 2:
                # Soft labels [batch_size, num_classes]
                # Expected positives: outer product of label distributions
                mask = torch.matmul(labels, labels.T).to(device)  # [batch_size, batch_size]
            else:
                raise ValueError("Labels must be 1D (hard) or 2D (soft)")
        else:
            mask = mask.float().to(device)

        contrast_count = n_views
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]     # only first view as anchor
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature   # all views are anchors
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Contrastive logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # for numerical stability (subtract the row-wise maximum value, does not change softmax probs)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast cases
        logits_mask = torch.ones_like(anchor_dot_contrast) - torch.eye(batch_size * anchor_count, device=device)
        mask = mask.repeat(anchor_count, contrast_count)
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        # Mean log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss