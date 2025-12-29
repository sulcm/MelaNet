import torch
import torch.nn.functional as F
import numpy as np

from typing import Union


def normalize_embeddings(embeds: Union[torch.Tensor, np.ndarray], eps: float = 1e-12) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize the embedding vectors to have unit length.

    Args:
        embeds: The raw embedding vectors

    Returns:
        norm_embeds: L2-normalized embedding vectors
    """
    if isinstance(embeds, torch.Tensor):
        return F.normalize(embeds, p=2, dim=-1, eps=eps)
    elif isinstance(embeds, np.ndarray):
        return embeds / (np.linalg.norm(embeds, axis=-1, keepdims=True) + eps)
    else:
        raise ValueError(f"Unsupported type of embedding: {type(embeds)}, must be `torch.Tensor` or `numpy.ndarray`")