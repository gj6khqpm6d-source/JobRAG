from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int | None = None) -> float:
    relevant = set(relevant_ids)
    limit = len(retrieved_ids) if k is None else min(k, len(retrieved_ids))
    for index, item_id in enumerate(retrieved_ids[:limit], start=1):
        if item_id in relevant:
            return 1 / index
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    gains = [1 if item_id in relevant else 0 for item_id in retrieved_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_length = min(k, len(relevant))
    ideal_dcg = sum(1 / math.log2(index + 2) for index in range(ideal_length))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def graded_ndcg_at_k(retrieved_ids: Sequence[str], relevance: dict[str, int], k: int) -> float:
    """nDCG with relevance grades such as 0 (irrelevant), 1 (partial), and 2 (high)."""
    gains = [relevance.get(item_id, 0) for item_id in retrieved_ids[:k]]
    dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted((gain for gain in relevance.values() if gain > 0), reverse=True)[:k]
    ideal_dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def condensed_graded_ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    """Evaluate ordering with incomplete judgments without treating unknown jobs as irrelevant.

    This is appropriate for the project's legacy annotation UI, where many
    unreviewed candidates were stored with the default zero value. Recall@K
    still penalizes missing known-positive jobs; condensed nDCG measures the
    ordering of the known graded positives that were retrieved.
    """
    positive_relevance = {item_id: grade for item_id, grade in relevance.items() if grade > 0}
    judged_ranking = [item_id for item_id in retrieved_ids if item_id in positive_relevance]
    return graded_ndcg_at_k(judged_ranking, positive_relevance, k)


def citation_coverage(cited_numbers: Sequence[int], claim_count: int) -> float:
    if claim_count <= 0:
        return 1.0
    return min(len(set(cited_numbers)) / claim_count, 1.0)
