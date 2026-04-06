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
            raise ValueError("Metric must be 'l2' or 'ip'")

        index.add(embs)

        return index

    def predict(self, query_embeddings: np.ndarray, top_k: int = 1, search_k: int = 20, return_scores: bool = False) -> list:
        """
        Retrieve top-K unique classes for each query embedding.
        Args:
            query_embeddings: np.ndarray of shape (Q, D)
            top_k: number of unique classes to return
            search_k: number of nearest neighbors to retrieve before filtering duplicates
            return_scores: return tuple per element with (class, score)
        Returns:
            List of lists of tuples [(class, score), ...] per query
        """
        if self.pca is not None:
            query_embeddings = self.pca(query_embeddings)
        distances, indices = self.index.search(query_embeddings, 1 if top_k == 1 else search_k)

        results = []
        for dist_row, idx_row in zip(distances, indices):
            if top_k == 1:
                cls = self.idx2cls[idx_row[0]]
                dist = dist_row[0]
                results.append(
                    (cls, dist) if return_scores else cls
                )
            else:
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
        return results