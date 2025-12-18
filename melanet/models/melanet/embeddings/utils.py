import torch
import numpy as np

from typing import Union


def normalize_embeddings(embeds: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize the embedding vectors to have unit length.

    Args:
        embeds: The raw embedding vectors

    Returns:
        norm_embeds: L2-normalized embedding vectors
    """
    if isinstance(embeds, torch.Tensor):
        return embeds / embeds.norm(dim=-1, keepdim=True)
    elif isinstance(embeds, np.ndarray):
        return embeds / np.linalg.norm(embeds, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unsupported type of embedding: {type(embeds)}, must be `torch.Tensor` or `numpy.ndarray`")