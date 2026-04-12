import faiss
import numpy as np

from typing import Optional
from collections import OrderedDict

from ..adapters import PCAAdapter


class NNClassifier():
    def __init__(self, cls_ids: list[int], embeddings: np.ndarray, metric: str = "ip", pca_components: Optional[int] = None):
        """
        Create a FAISS index for nearest-neighbor classification.
        Args:
            embedding_dim: Dimension of the embedding vectors.
            metric: 'l2' for Euclidean or 'ip' for inner product (cosine similarity if normalized).
        """
        self.metric = metric
        self.pca: Optional[PCAAdapter] = None

        self.idx2cls = cls_ids
        self.index = self._build_index(embeddings, metric, pca_components=pca_components)

    def _build_index(self, embs: np.ndarray, metric: str, pca_components: Optional[int] = None):
        if pca_components is not None:
            self.pca = PCAAdapter(out_features=pca_components, whiten=True, normalize=True)
            self.pca = self.pca.fit(embs)
        if self.pca is not None:
            embs = self.pca(embs)

        if metric == "l2":
            index = faiss.IndexFlatL2(embs.shape[1])
        elif metric == "ip":
            index = faiss.IndexFlatIP(embs.shape[1])
        else:
            raise ValueError(f"Metric must be 'l2' or 'ip' but you have provided '{metric}'")

        index.add(embs)

        return index

    def predict(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 1,
        search_k: int = 20,
        unique_only: bool = False,
        return_scores: bool = False
    ) -> list[int | tuple[int, float]]:
        """
        Retrieve classes for each query embedding.

        Args:
            query_embeddings (np.ndarray): Numpy ndarray of shape (Q, D).
            top_k (int): Number of predictions (classifications) to return. Defaults to `1`.
            search_k (int): Number of nearest neighbors to retrieve before filtering. Defaults to `20`.
            unique_only (bool): Return only unique (with best score) representants to given query (applies only when `top_k > 1`). Defaults to `False`.
            return_scores (bool): Return tuple per element as `(class, score)`. Defaults to `False`.
        Returns:
            list[int | tuple[int, float]]: List of predictions `[class, ...]` or tuples `[(class, score), ...]` per query.
        """
        if self.pca is not None:
            query_embeddings = self.pca(query_embeddings)

        if top_k == 1:
            __search_k = 1
        else:
            if not unique_only:
                __search_k = top_k
            else:
                __search_k = search_k

        distances, indices = self.index.search(query_embeddings, __search_k)

        results = []
        for dist_row, idx_row in zip(distances, indices):
            if top_k == 1:
                cls = self.idx2cls[idx_row[0]]
                dist = dist_row[0]
                results.append(
                    (cls, dist) if return_scores else cls
                )
            else:
                if unique_only:
                    seen = OrderedDict()
                    for d, idx in zip(dist_row, idx_row):
                        cls = self.idx2cls[idx]
                        if cls not in seen:
                            seen[cls] = d
                        if len(seen) >= top_k:
                            break
                    results.append(
                        list(seen.items()) if return_scores else list(seen.keys())
                    )
                else:
                    results_top_k = []
                    for d, idx in zip(dist_row, idx_row):
                        cls = self.idx2cls[idx]
                        results_top_k.append(
                            (cls, d) if return_scores else cls
                        )
                        if len(results_top_k) >= top_k:
                            break
                    results.append(results_top_k)

        return results