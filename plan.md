# Build Plan — Recsys Studio

## Interview thesis
"I evaluate recommenders like a production team: implicit feedback, a **temporal** split, and
**ranking** metrics (NDCG/Recall/MAP) — not RMSE on a random split. I can show that a strong
popularity baseline is hard to beat, and where matrix factorization actually wins." Ranking +
leakage awareness is the senior signal most recsys demos miss.

## Architecture
```
data.py    load MovieLens; implicit matrix (rating>=4); TEMPORAL train/test split
models.py  Popularity, ItemKNN (cosine on item vectors), ALS/MF — common recommend(user, k)
metrics.py NDCG@k, Recall@k, MAP@k, catalog coverage, novelty
eval.py    rank top-k per user on the held-out future -> eval/results.md
app.py     Streamlit: user picker -> top-k recs + explanation + leaderboard
```

## Evaluation (defend each)
- **Temporal split:** order interactions by timestamp; train on the past, test on each user's later
  positives. A random split leaks future co-occurrence — the #1 recsys eval mistake.
- **Ranking metrics:** NDCG@k (position-weighted), Recall@k (did we surface their held-out items),
  MAP@k. RMSE is deliberately *not* the headline.
- **Beyond accuracy:** **coverage** (how much of the catalog is ever recommended) and **novelty**
  (are we just pushing blockbusters?) — guards against a degenerate "recommend top-10 popular".
- **Cold-start:** report metrics separately for low-history users; popularity often wins there.

## Key decisions
- **Implicit, not explicit:** binarize ratings; this matches real clickstream/purchase data.
- **Strong baseline:** popularity is the bar every model must clear — stated up front.
- **Explanation:** item-kNN gives "because you liked X" for free — a nice UI + interview point.

## Phases (≈12 days, 2–3 h/day)
1. **Data + temporal split** — implicit matrix; frozen time-based holdout; leakage check.
2. **Baselines** — popularity + item-item kNN behind a common `recommend(user, k)`.
3. **Matrix factorization / ALS** — `implicit` if it builds, else NumPy ALS; same interface.
4. **Ranking metrics** — NDCG/Recall/MAP + coverage/novelty; verify on a tiny hand-checked case.
5. **App + README** — per-user recs + leaderboard; copy real numbers in.

## Py3.14 de-risk
`implicit` is Cython and may not have a 3.14 wheel — **test the install on day 1**. Fallback: a
~50-line NumPy/scipy ALS (or `TruncatedSVD` on the sparse matrix) + item-kNN; the *evaluation
framework* is the project, the specific MF library is swappable. No torch.

## Deploy
Streamlit Cloud or HF Spaces. Bundle a MovieLens-100k sample (smaller) so the demo loads fast
without the full download. Steps in `ManualSteps.md`.
