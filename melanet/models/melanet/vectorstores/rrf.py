from collections import defaultdict


def reciprocal_rank_fusion(results: list[list], top_n: int = 1, k: float = 60.0, items_have_scores: bool = False) -> list:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion (RRF).
    
    Args:
        results (list[list]): List of ranked lists, e.g. [[id1, id2, id3], [id2, id1, id4], ...]
        top_n (int): Number of top items to return.
        k (int): RRF constant controlling decay of rank influence.

    Returns:
        list[tuple]: List of item_ids sorted by fused RRF score descending.
    """
    scores = defaultdict(float)
    if items_have_scores:
        for ranked_list in results:
            for rank, (item_id, item_score) in enumerate(ranked_list, start=1):
                scores[item_id] += 1.0 / (k + rank)
    else:
        for ranked_list in results:
            for rank, item_id in enumerate(ranked_list, start=1):
                scores[item_id] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [rrf_item for rrf_item, score in fused[:top_n]]


def melanet_rrf(retrieved_results: dict[str, list[list]], top_n: int = 1, k: float = 60.0) -> list:
    results = []
    for results2rerank in zip(*retrieved_results.values()):
        rrf_pred = reciprocal_rank_fusion(
            results=results2rerank,
            top_n=top_n,
            k=k,
            items_have_scores=isinstance(results2rerank[0][0], (list, tuple))
        )
        if top_n == 1:
            rrf_pred = rrf_pred[0]
        results.append(rrf_pred)
    return results