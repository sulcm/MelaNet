import numpy as np

from typing import Optional, Union

from .nn_classifier import NNClassifier


class MultiNNClassifier():
    default_metric = "ip"
    default_pca_components = None

    def __init__(
        self,
        cls_ids: list[int],
        embeddings: dict[str, np.ndarray],
        metric: Union[list[str], str] = None,
        pca_components: Union[list[Optional[int]], Optional[int]] = None,
        l2_normalize: bool = False
    ):
        if metric is None:
            metric = [self.default_metric,] * len(embeddings)
        elif isinstance(metric, str):
            metric = [metric,] * len(embeddings)
        assert len(metric) == len(embeddings), "You must provide same number of metrics as is number of initialized vector stores or `None`."

        if pca_components is None:
            pca_components = [self.default_pca_components,] * len(embeddings)
        elif isinstance(pca_components, int):
            pca_components = [pca_components,] * len(embeddings)
        assert len(pca_components) == len(embeddings), "You must provide same number of PCA components as is number of initialized vector stores or `None`."

        self.l2_normalize = l2_normalize
        self.idx2cls = cls_ids
        self.indexes = {
            _idx_name: NNClassifier(
                cls_ids=cls_ids,
                embeddings=_embeds,
                metric=_metric,
                pca_components=_pca_c,
                l2_normalize=l2_normalize
            )
            for (_idx_name, _embeds), _metric, _pca_c in zip(embeddings.items(), metric, pca_components)
        }

    def predict(
        self,
        query_embeddings: dict[str, np.ndarray],
        top_k: int = 1,
        search_k: int = 20,
        unique_only: bool = False,
        return_scores: bool = False
    ) -> dict[str, list[int | tuple[int, float]]]:
        """
        Retrieve classes for each query embedding from every specified index.

        Args:
            query_embeddings (dict[str, np.ndarray]): Index names to search with queries (Numpy ndarray of shape (Q, D)).
            top_k (int): Number of predictions (classifications) to return. Defaults to `1`.
            search_k (int): Number of nearest neighbors to retrieve before filtering. Defaults to `20`.
            unique_only (bool): Return only unique (with best score) representants to given query (applies only when `top_k > 1`). Defaults to `False`.
            return_scores (bool): Return tuple per element as `(class, score)`. Defaults to `False`.
        Returns:
            dict[str, list[int | tuple[int, float]]]: Per index results as list of predictions `[class, ...]` or tuples `[(class, score), ...]` per query.
        """
        search_results = {}
        for index_name, query_embed in query_embeddings.items():
            search_results[index_name] = self.indexes[index_name].predict(
                query_embeddings=query_embed,
                top_k=top_k,
                search_k=search_k,
                unique_only=unique_only,
                return_scores=return_scores
            )
        return search_results