import numpy as np

from typing import Optional

from .nn_classifier import NNClassifier


class MultiNNClassifier():
    default_metric = "ip"
    default_pca_components = None

    def __init__(
        self,
        cls_ids: list[int],
        embeddings: dict[str, np.ndarray],
        metric: list[str] = None,
        pca_components: list[Optional[int]] = None
    ):
        if metric is None:
            metric = [self.default_metric,] * len(embeddings)
        if pca_components is None:
            pca_components = [self.default_pca_components,] * len(embeddings)
        assert len(metric) == len(embeddings), "You must provide same number of metrics as is number of initialized vector stores or `None`."
        assert len(pca_components) == len(embeddings), "You must provide same number of PCA components as is number of initialized vector stores or `None`."

        self.idx2cls = cls_ids
        self.indexes = {
            _idx_name: NNClassifier(
                cls_ids=cls_ids,
                embeddings=_embeds,
                metric=_metric,
                pca_components=_pca_c
            )
            for (_idx_name, _embeds), _metric, _pca_c in zip(embeddings.items(), metric, pca_components)
        }

    def predict(self, query_embeddings: dict[str, np.ndarray], top_k: int = 1, search_k: int = 20, return_scores: bool = False) -> dict[str, list]:
        """
        Retrieve top-K unique classes for each query embedding.
        Args:
            query_embeddings: dict of index names to search with queries (np.ndarray of shape (Q, D))
            top_k: number of unique classes to return
            search_k: number of nearest neighbors to retrieve before filtering duplicates
            return_scores: return tuple per element with (class, score)
        Returns:
            Dict with lists of results per index
        """
        search_results = {}
        for index_name, query_embed in query_embeddings.items():
            search_results[index_name] = self.indexes[index_name].predict(
                query_embeddings=query_embed,
                top_k=top_k,
                search_k=search_k,
                return_scores=return_scores
            )
        return search_results