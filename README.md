# 🎬 Recsys Studio — recommendations judged by *ranking*, not RMSE

An implicit-feedback recommender benchmarked the way real recsys teams actually evaluate: **ranking
metrics** (NDCG@k, Recall@k, MAP) on a **leakage-free temporal split**, with coverage and novelty so
the leaderboard isn't gamed by always recommending blockbusters.

> **Status:** scaffolding (plan + schedule below). Results table is a template filled only from the
> real eval run — no numbers until measured. Mirrors the backtesting rigor of
> [Forecasting Studio](https://github.com/PruthviVKadam/forecast-studio) (P4), applied to ranking.

## Problem → Approach → Result

- **Problem:** most portfolio recommenders report **RMSE on a random split** — which rewards
  predicting ratings nobody ranks by, and leaks the future into the past. Production recsys cares
  about *the order of the top-k* and whether it works on **tomorrow's** interactions.
- **Approach:** treat MovieLens as **implicit feedback** (rating ≥ 4 = positive), split **by time**
  (train on the past, evaluate on each user's later interactions), and compare **popularity →
  item-item kNN → matrix factorization / ALS**. Score with **NDCG@k, Recall@k, MAP@k** plus
  **catalog coverage** and **novelty**. A Streamlit app: pick a user, see recs + *why*, and the
  metric leaderboard.
- **Result:** _(filled from `eval.py` output — no numbers until measured)_

| Model | NDCG@10 | Recall@10 | Coverage |
| --- | --- | --- | --- |
| Popularity (baseline) | _TBD_ | _TBD_ | _TBD_ |
| Item-item kNN | _TBD_ | _TBD_ | _TBD_ |
| ALS / MF | _TBD_ | _TBD_ | _TBD_ |

## Dataset
**MovieLens-1M** (~1M ratings, 6k users, 4k movies; public, no auth). Optional: **RetailRocket**
(real implicit e-commerce events) for a "true implicit, no explicit ratings" story.

## Stack

Python 3.14 · scipy sparse · numpy · scikit-learn · Plotly · Streamlit · **`implicit`** (ALS) *if it
builds on 3.14; otherwise a NumPy/scipy MF + item-kNN fallback* (see `plan.md`).

## Reproduce (once built)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python data.py        # load MovieLens, build implicit matrix, TEMPORAL split
python train.py       # popularity, item-kNN, ALS/MF
python eval.py        # NDCG@k / Recall@k / MAP / coverage / novelty -> eval/results.md
streamlit run app.py  # per-user recs + leaderboard
```

## Honesty guardrail

The leaderboard is copied verbatim from `eval/results.md`. The split is **temporal** (documented), so
no metric benefits from seeing the future; a model that can't beat the popularity baseline is shown
losing — the same "baselines are hard to beat" honesty as P4.
