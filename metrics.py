"""Ranking metrics (NDCG / Recall / MAP) plus beyond-accuracy metrics (coverage, novelty).

Why ranking and not RMSE: a deployed recommender shows an *ordered* top-k; users act on
the first few slots. RMSE rewards predicting a 3.8 vs a 4.0 -- a number nobody ranks by.
These metrics score the order of the list against each user's held-out future positives.

All functions take `recommended` (an ordered list/array of item ids, best first) and
`relevant` (the set of held-out positive item ids for that user).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np


def _log2(x: int) -> float:
    return math.log2(x)


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Position-weighted: a hit at rank 1 is worth more than a hit at rank k.
    Normalized by the best achievable DCG given how many relevant items exist."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / _log2(i + 2)  # rank i (0-based) -> discount log2(i+2)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / _log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """AP@k: mean of precision@rank taken at each relevant hit, normalized by min(k,|rel|).
    Rewards packing relevant items high AND early. MAP is the mean of this over users."""
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(k, len(relevant))


def coverage(all_recommended: Iterable[Sequence[int]], n_items: int) -> float:
    """Catalog coverage: fraction of the catalog that ever appears in any user's top-k.
    Guards against a model that only ever surfaces the same few blockbusters."""
    surfaced: set[int] = set()
    for rec in all_recommended:
        surfaced.update(rec)
    return len(surfaced) / n_items


def novelty(all_recommended: Iterable[Sequence[int]],
            item_popularity: np.ndarray, n_users: int) -> float:
    """Mean self-information (bits) of recommended items: -log2(P(item)).
    Popular items carry little information (low novelty); long-tail items carry more.
    A high-novelty list is not automatically better, but a *very low* one means the model
    is just re-serving the head of the catalog."""
    info = 0.0
    count = 0
    for rec in all_recommended:
        for item in rec:
            p = item_popularity[item] / n_users
            if p > 0:
                info += -_log2(p)
                count += 1
    return info / count if count else 0.0


def evaluate(model, dataset, k: int = 10) -> dict[str, float]:
    """Score a fitted model over every test user on the temporal holdout."""
    ndcgs, recalls, aps = [], [], []
    all_recs: list[np.ndarray] = []
    for user_idx, relevant in dataset.test.items():
        rec = model.recommend(user_idx, k=k, exclude_seen=True)
        ndcgs.append(ndcg_at_k(rec, relevant, k))
        recalls.append(recall_at_k(rec, relevant, k))
        aps.append(average_precision_at_k(rec, relevant, k))
        all_recs.append(rec)
    return {
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"MAP@{k}": float(np.mean(aps)),
        "Coverage": coverage(all_recs, dataset.n_items),
        "Novelty(bits)": novelty(all_recs, dataset.item_popularity, dataset.n_users),
        "n_users": len(dataset.test),
    }
