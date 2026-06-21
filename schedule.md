# Build Schedule — Recsys Studio (~12 days, 2–3 h/day)

| Day | Focus | Done when |
| --- | --- | --- |
| 1 | Repo, venv; **test `implicit` install on Py3.14**; load MovieLens | sparse matrix built |
| 2 | Implicit binarization (rating≥4) + **temporal** split + leakage check | holdout frozen; no future leak |
| 3 | Popularity baseline behind `recommend(user, k)` | baseline recs for any user |
| 4 | Item-item kNN (cosine) + "because you liked X" explanation | neighbor recs + reasons |
| 5 | ALS/MF (implicit lib or NumPy fallback) | MF recs via same interface |
| 6 | NDCG@k + Recall@k | metrics verified on a hand-checked toy case |
| 7 | MAP@k + coverage + novelty | full metric set runs |
| 8 | `eval.py` leaderboard over all models | `eval/results.md` produced |
| 9 | Cold-start breakdown (low-history users) | separate metrics reported |
| 10 | Streamlit app: user picker, recs, leaderboard | app runs locally |
| 11 | pytest (metrics + temporal-split guard) + README real numbers | tests green; README filled |
| 12 | Deploy + 2-min walkthrough | live URL; recording done |

**Lead-with-the-number (resume line, once real):**
"NDCG@10 of __ (vs __ for a popularity baseline) on a **temporal** MovieLens split — with __%
catalog coverage, evaluated the way production recsys teams actually measure ranking."

**Cut scope if behind:** drop ALS and novelty; popularity + item-kNN + NDCG/Recall + temporal split
+ the app is a complete, honest project.
