"""Load MovieLens-100k, binarize to implicit feedback, and build a LEAKAGE-FREE
temporal split.

Design decisions (each defensible in an interview):
  * Implicit feedback  -> a rating >= POSITIVE_THRESHOLD (4) is a "positive"; everything
    else is treated as unobserved. This matches real click/purchase data, where you only
    see what users engaged with, never an explicit dislike.
  * GLOBAL temporal split -> pick one cutoff timestamp (a quantile of all interactions).
    Train = positives strictly before the cutoff, test = positives at/after it. By
    construction every training event precedes every test event, so NO future
    co-occurrence can leak into the past. A random split leaks the future and is the #1
    recsys evaluation mistake.
  * The item/user vocabulary is taken from the TRAIN window only. A model cannot
    recommend an item it never saw in training; test positives on cold items are dropped
    from scoring (and that loss is honest, not hidden) -- popularity has the same limit.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR = pathlib.Path(__file__).parent / "data"
POSITIVE_THRESHOLD = 4
CUTOFF_QUANTILE = 0.80  # 80% of interactions (by time) train, last 20% test


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ratings.csv")


def load_movie_titles() -> dict[int, str]:
    m = pd.read_csv(DATA_DIR / "movies.csv")
    return dict(zip(m.item_id, m.title))


@dataclass
class Dataset:
    train: sp.csr_matrix                # users x items, 1.0 for a train positive
    test: dict[int, set[int]]           # user_idx -> held-out positive item_idx (future)
    user_ids: np.ndarray                # original MovieLens user ids, indexed by user_idx
    item_ids: np.ndarray                # original MovieLens item ids, indexed by item_idx
    item_popularity: np.ndarray         # train positive count per item_idx
    cutoff_ts: int
    n_train_events: int
    n_test_events: int

    @property
    def n_users(self) -> int:
        return self.train.shape[0]

    @property
    def n_items(self) -> int:
        return self.train.shape[1]

    def seen_items(self, user_idx: int) -> np.ndarray:
        """Item indices this user already interacted with in TRAIN (to exclude from recs)."""
        return self.train[user_idx].indices


def build_dataset(cutoff_quantile: float = CUTOFF_QUANTILE,
                  threshold: int = POSITIVE_THRESHOLD) -> Dataset:
    df = load_ratings()
    cutoff = int(df.timestamp.quantile(cutoff_quantile))

    before = df[df.timestamp < cutoff]
    after = df[df.timestamp >= cutoff]

    train_pos = before[before.rating >= threshold]
    test_pos = after[after.rating >= threshold]

    # Leakage guard: every training event strictly precedes the cutoff and every test
    # event is at/after it -> the maximum train time is below the minimum test time.
    assert train_pos.timestamp.max() < test_pos.timestamp.min(), "temporal leak!"

    # Vocabulary from the train window only.
    user_ids = np.sort(train_pos.user_id.unique())
    item_ids = np.sort(train_pos.item_id.unique())
    u2idx = {u: k for k, u in enumerate(user_ids)}
    i2idx = {it: k for k, it in enumerate(item_ids)}

    rows = train_pos.user_id.map(u2idx).to_numpy()
    cols = train_pos.item_id.map(i2idx).to_numpy()
    train = sp.csr_matrix(
        (np.ones(len(train_pos), dtype=np.float32), (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )
    train.data[:] = 1.0  # collapse any duplicate (user,item) to a single positive
    train.sum_duplicates()
    train.data[:] = 1.0

    item_popularity = np.asarray(train.sum(axis=0)).ravel()

    # Test set: keep only users and items the model could have learned in train.
    test_pos = test_pos[test_pos.user_id.isin(u2idx) & test_pos.item_id.isin(i2idx)]
    test: dict[int, set[int]] = {}
    for user_id, grp in test_pos.groupby("user_id"):
        uidx = u2idx[user_id]
        held = {i2idx[it] for it in grp.item_id} - set(train[uidx].indices)
        if held:  # only score users who have at least one *new* future positive
            test[uidx] = held

    return Dataset(
        train=train,
        test=test,
        user_ids=user_ids,
        item_ids=item_ids,
        item_popularity=item_popularity,
        cutoff_ts=cutoff,
        n_train_events=int(train.nnz),
        n_test_events=int(sum(len(v) for v in test.values())),
    )


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ds = build_dataset()
    print("MovieLens-100k  |  implicit feedback (rating >= 4)")
    print(f"  global temporal cutoff ts : {ds.cutoff_ts}")
    print(f"  train users x items       : {ds.n_users} x {ds.n_items}")
    print(f"  train positive events     : {ds.n_train_events}")
    print(f"  test users (scored)       : {len(ds.test)}")
    print(f"  test positive events      : {ds.n_test_events}")
    dens = ds.n_train_events / (ds.n_users * ds.n_items)
    print(f"  train matrix density      : {dens:.4f}")
