"""Tests: a hand-verified golden case for the metrics, and structural guarantees for the
recommenders + the temporal split (the things that, if wrong, silently inflate scores)."""
import numpy as np
import scipy.sparse as sp

from data import build_dataset
from metrics import average_precision_at_k, coverage, ndcg_at_k, recall_at_k
from models import ItemKNNRecommender, PopularityRecommender


# ---- Golden metric case (computed by hand in the docstring of each assert) ----------------
REC = [10, 20, 30, 40]          # ranked recommendations (best first)
RELEVANT = {20, 40}             # held-out positives at ranks 2 and 4


def test_ndcg_golden():
    # dcg = 1/log2(3) + 1/log2(5) = 0.6309298 + 0.4306766 = 1.0616064
    # idcg (2 relevant) = 1/log2(2) + 1/log2(3) = 1 + 0.6309298 = 1.6309298
    # ndcg = 1.0616064 / 1.6309298 = 0.6509209
    assert abs(ndcg_at_k(REC, RELEVANT, 4) - 0.6509209) < 1e-6


def test_recall_golden():
    # both relevant items are in the top-4 -> 2/2
    assert recall_at_k(REC, RELEVANT, 4) == 1.0


def test_map_golden():
    # hit@rank2 -> 1/2 ; hit@rank4 -> 2/4 ; sum 1.0 ; / min(4,2)=2 -> 0.5
    assert average_precision_at_k(REC, RELEVANT, 4) == 0.5


def test_perfect_ranking_is_one():
    assert ndcg_at_k([20, 40, 99], RELEVANT, 3) == 1.0
    assert recall_at_k([20, 40, 99], RELEVANT, 3) == 1.0


def test_empty_relevant_is_zero():
    assert ndcg_at_k(REC, set(), 4) == 0.0
    assert recall_at_k(REC, set(), 4) == 0.0
    assert average_precision_at_k(REC, set(), 4) == 0.0


def test_ndcg_rewards_higher_rank():
    # same hit, but earlier -> strictly higher NDCG
    early = ndcg_at_k([20, 1, 2, 3], {20}, 4)
    late = ndcg_at_k([1, 2, 3, 20], {20}, 4)
    assert early > late


def test_coverage_fraction():
    assert coverage([[1, 2], [2, 3]], n_items=10) == 3 / 10


# ---- Recommender guarantees ---------------------------------------------------------------
def _toy_matrix():
    # 3 users x 4 items; user0 likes items 0,1 ; user1 likes 1,2 ; user2 likes 0,3
    rows = [0, 0, 1, 1, 2, 2]
    cols = [0, 1, 1, 2, 0, 3]
    return sp.csr_matrix((np.ones(6), (rows, cols)), shape=(3, 4))


def test_recommend_excludes_seen():
    m = _toy_matrix()
    rec = PopularityRecommender().fit(m)
    out = rec.recommend(0, k=4, exclude_seen=True)
    assert 0 not in out and 1 not in out  # user0 already saw items 0 and 1


def test_recommend_respects_k():
    m = _toy_matrix()
    rec = ItemKNNRecommender(topk_neighbors=3).fit(m)
    assert len(rec.recommend(0, k=2)) <= 2


def test_popularity_orders_by_count():
    m = _toy_matrix()
    # global counts: item1=2, item0=2, item2=1, item3=1
    rec = PopularityRecommender().fit(m)
    # for user1 (saw 1,2), top unseen should be item0 (count 2) before item3 (count 1)
    out = list(rec.recommend(1, k=4))
    assert out[0] == 0


# ---- Temporal split leakage guard ---------------------------------------------------------
def test_temporal_split_has_no_leak():
    ds = build_dataset()
    # every held-out test item must be NEW to that user (never in their train history)
    for user_idx, relevant in ds.test.items():
        seen = set(ds.seen_items(user_idx))
        assert relevant.isdisjoint(seen), "a test positive was already in train!"
    assert ds.n_test_events > 0 and len(ds.test) > 0
