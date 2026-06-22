---
title: Recsys Studio
emoji: 🎬
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# 🎬 Recsys Studio — recommendations judged by *ranking*, not RMSE

An implicit-feedback recommender benchmarked the way real recsys teams actually evaluate: **ranking
metrics** (NDCG@10, Recall@10, MAP) on a **leakage-free temporal split**, with coverage and novelty so
the leaderboard isn't gamed by always recommending blockbusters.

**▶️ Live demo:** https://pk759-recsys-studio.hf.space  ·  **Code:** https://github.com/PruthviVKadam/recsys-studio

## Problem → Approach → Result

- **Problem:** most portfolio recommenders report **RMSE on a random split** — which rewards
  predicting ratings nobody ranks by, and leaks the future into the past. Production recsys cares
  about *the order of the top-k* and whether it works on **tomorrow's** interactions.
- **Approach:** treat MovieLens-100k as **implicit feedback** (rating ≥ 4 = positive), split **by
  time at one global cutoff** (first 80% of interactions = train, last 20% = test), and compare
  **Popularity → item-item kNN → ALS matrix factorization** behind a single `recommend(user, k)`
  interface. Score with **NDCG@10, Recall@10, MAP@10** plus **catalog coverage** and **novelty**.
  A Streamlit app: pick a user, see recs + *why*, and the metric leaderboard.
- **Result** — scored on **94 test users / 1,545 held-out future positives** (copied verbatim from
  [`eval/results.md`](eval/results.md)):

| Model | NDCG@10 | Recall@10 | MAP@10 | Coverage | Novelty (bits) |
| --- | --- | --- | --- | --- | --- |
| **Popularity** (baseline) | **0.1236** | **0.0871** | 0.0732 | 0.039 | 1.64 |
| Item-item kNN | 0.1196 | 0.0483 | **0.0776** | 0.083 | 2.04 |
| ALS / MF | 0.1034 | 0.0542 | 0.0581 | **0.295** | **3.25** |

The strong popularity baseline **wins raw accuracy** (NDCG, Recall); ItemKNN edges it on MAP; ALS
trails on accuracy but covers **7.5× more of the catalog** (29.5% vs 3.9%) at the highest novelty.
No model was tuned to make a favorite win — these are the honest numbers.

## 💡 Insights gained

- **A non-personalized baseline is genuinely hard to beat — and most demos never check.** On a
  *strict temporal* split, recommending the globally most-liked unseen movies scores the best
  NDCG@10 (0.1236) and Recall@10 (0.0871). Any claim that "my matrix factorization works" is
  meaningless until it clears this bar; here MF (ALS) does **not** clear it on accuracy. Reporting
  that honestly is the senior signal.
- **"Best" depends on the metric — accuracy vs. coverage/novelty is a real trade.** ALS is last on
  accuracy yet first on **coverage (0.295)** and **novelty (3.25 bits)**: it spreads recommendations
  across the catalog instead of re-serving the head. A product team weighing engagement vs.
  catalog discovery would pick differently than an offline-accuracy leaderboard suggests.
- **How you split the data changes the story more than which model you pick.** The *global* temporal
  cutoff leaves only **94 test users** (most MovieLens users do all their rating in one burst, so
  they fall entirely on one side of the cutoff). A per-user split would show ~900 users and prettier
  numbers — but it re-introduces the cross-user future co-occurrence leak this project exists to
  avoid. I chose the smaller honest number over the bigger leaky one.
- **Item-item CF gives explanations for free, which matters for trust.** Because a kNN score is a sum
  of similarities to items you already liked, every recommendation comes with a "because you liked
  *Apollo 13*" reason — something MF latent factors can't produce without extra work.
- **Cold-start nuance:** splitting test users by train-history (median = 89 liked items), ItemKNN
  actually **leads on lighter-history users** (NDCG@10 0.1609 vs Popularity's 0.1525), while
  Popularity leads on heavy-history users — the opposite of the usual "popularity wins cold-start"
  assumption, and only visible because the eval breaks it out.

## Dataset

**MovieLens-100k** (100,000 ratings · 943 users · 1,682 movies; public, no auth). Bundled as small
CSVs (`data/ratings.csv`, `data/movies.csv`, ~2 MB) so the demo loads instantly. The raw `.zip` is
git-ignored; see `ManualSteps.md` to regenerate.

## Stack

Python 3.14 · `implicit` 0.7.3 (ALS) · scipy sparse · scikit-learn (cosine) · pandas · Plotly ·
Streamlit. `implicit` installs as a prebuilt wheel on 3.14 — no Cython build, no fallback needed.

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python data.py        # build implicit matrix + global TEMPORAL split (prints split stats)
python eval.py        # NDCG/Recall/MAP + coverage/novelty -> eval/results.md
pytest -q             # 11 tests: hand-verified metric golden case + leakage/guard properties
streamlit run app.py  # per-user recs + "because you liked X" + leaderboard
```

## Honesty guardrail

The leaderboard is copied verbatim from `eval/results.md`. The split is **global temporal**
(documented), so no metric benefits from seeing the future; a model that can't beat the popularity
baseline is shown losing — the same "baselines are hard to beat" discipline as
[Forecasting Studio](https://github.com/PruthviVKadam/forecast-studio) (P4). The NDCG/Recall/MAP
formulas are pinned by a hand-computed golden test in `tests/`.
