"""Three recommenders behind one interface, from a hard baseline to matrix factorization.

    Popularity  -> the bar every model must clear (recommend the globally most-liked items).
    ItemKNN     -> item-item collaborative filtering (cosine); gives "because you liked X".
    ALS         -> implicit-feedback matrix factorization (the `implicit` library).

Every model exposes `score_all(user_idx) -> np.ndarray` over the full item catalog, so a
single `recommend()` handles top-k selection and excludes items the user already saw in
train. That keeps the evaluation identical across models -- the only thing that changes is
how the scores are produced.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


class BaseRecommender:
    name = "base"

    def fit(self, train: sp.csr_matrix) -> "BaseRecommender":
        self.train = train.tocsr()
        return self

    def score_all(self, user_idx: int) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def recommend(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        scores = self.score_all(user_idx).astype(np.float64)
        if exclude_seen:
            scores[self.train[user_idx].indices] = -np.inf
        # argpartition for the top-k, then sort just those k by score (descending)
        k = min(k, np.isfinite(scores).sum())
        if k <= 0:
            return np.array([], dtype=int)
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])]


class PopularityRecommender(BaseRecommender):
    """Non-personalized: every user gets the most popular unseen items. The honest bar."""

    name = "Popularity"

    def fit(self, train: sp.csr_matrix) -> "PopularityRecommender":
        super().fit(train)
        self.popularity = np.asarray(train.sum(axis=0)).ravel()
        return self

    def score_all(self, user_idx: int) -> np.ndarray:
        return self.popularity


class ItemKNNRecommender(BaseRecommender):
    """Item-based CF. Similarity = cosine between item columns of the train matrix.
    A user's score for item i is the summed similarity of i to the items they liked,
    which is exactly what powers a 'because you liked X' explanation."""

    name = "ItemKNN"

    def __init__(self, topk_neighbors: int = 50):
        self.topk_neighbors = topk_neighbors

    def fit(self, train: sp.csr_matrix) -> "ItemKNNRecommender":
        super().fit(train)
        # items x users, cosine across users -> items x items similarity
        sim = cosine_similarity(train.T, dense_output=True).astype(np.float32)
        np.fill_diagonal(sim, 0.0)  # an item never recommends itself
        if self.topk_neighbors and self.topk_neighbors < sim.shape[0]:
            # keep only each item's strongest neighbors (denoise + speed)
            for i in range(sim.shape[0]):
                row = sim[i]
                drop = np.argpartition(-row, self.topk_neighbors)[self.topk_neighbors:]
                row[drop] = 0.0
        self.sim = sim
        return self

    def score_all(self, user_idx: int) -> np.ndarray:
        seen = self.train[user_idx].indices
        if len(seen) == 0:
            return np.zeros(self.sim.shape[0], dtype=np.float32)
        return self.sim[seen].sum(axis=0)

    def explain(self, user_idx: int, item_idx: int) -> int | None:
        """Which already-liked item contributes most to recommending `item_idx`."""
        seen = self.train[user_idx].indices
        if len(seen) == 0:
            return None
        contrib = self.sim[item_idx, seen]
        if contrib.max() <= 0:
            return None
        return int(seen[int(np.argmax(contrib))])


class ALSRecommender(BaseRecommender):
    """Implicit-feedback matrix factorization via the `implicit` library's ALS."""

    name = "ALS"

    def __init__(self, factors: int = 64, iterations: int = 20,
                 regularization: float = 0.05, use_bm25: bool = True,
                 random_state: int = 42):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.use_bm25 = use_bm25  # BM25 confidence weighting is implicit's documented default
        self.random_state = random_state

    def fit(self, train: sp.csr_matrix) -> "ALSRecommender":
        super().fit(train)
        from implicit.als import AlternatingLeastSquares
        from implicit.nearest_neighbours import bm25_weight

        conf = train
        if self.use_bm25:
            # down-weight popular items, scale confidence by interaction strength
            conf = bm25_weight(train, K1=1.2, B=0.75).tocsr().astype(np.float32)

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
            use_gpu=False,
        )
        # implicit >=0.5 fits a user-items confidence matrix
        self.model.fit(conf, show_progress=False)
        return self

    def score_all(self, user_idx: int) -> np.ndarray:
        return self.model.user_factors[user_idx] @ self.model.item_factors.T


MODELS = {
    "Popularity": PopularityRecommender,
    "ItemKNN": ItemKNNRecommender,
    "ALS": ALSRecommender,
}
