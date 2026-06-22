"""Recsys Studio — pick a user, see their top-k recommendations from each model (with a
'because you liked X' explanation), and the leaderboard scored on a leakage-free temporal
split. Numbers here are computed live from the same code as eval.py."""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
import plotly.express as px
import streamlit as st

from data import build_dataset, load_movie_titles
from metrics import evaluate
from models import ALSRecommender, ItemKNNRecommender, PopularityRecommender

st.set_page_config(page_title="Recsys Studio", page_icon="🎬", layout="wide")
K = 10


@st.cache_resource(show_spinner="Building temporal split + fitting models…")
def load():
    ds = build_dataset()
    models = {
        "Popularity": PopularityRecommender().fit(ds.train),
        "ItemKNN": ItemKNNRecommender(topk_neighbors=50).fit(ds.train),
        "ALS": ALSRecommender(factors=64, iterations=20).fit(ds.train),
    }
    titles = load_movie_titles()
    leaderboard = {name: evaluate(m, ds, k=K) for name, m in models.items()}
    return ds, models, titles, leaderboard


ds, MODELS, TITLES, LEADERBOARD = load()


def title(item_idx: int) -> str:
    return TITLES.get(int(ds.item_ids[item_idx]), f"item {ds.item_ids[item_idx]}")


st.title("🎬 Recsys Studio")
st.caption("Implicit-feedback recommenders judged by **ranking** (NDCG / Recall / MAP) on a "
           "**leakage-free temporal split** — not RMSE on a random split.")

tab_try, tab_board = st.tabs(["▶️  Try the recommender", "🏆  Leaderboard & method"])

# ----------------------------------------------------------------------------- Try it ------
with tab_try:
    test_users = sorted(ds.test.keys())
    c1, c2 = st.columns([1, 1])
    with c1:
        uidx = st.selectbox(
            "User (showing users with held-out future positives, so you can see hits)",
            test_users,
            format_func=lambda u: f"user {ds.user_ids[u]}  ·  {len(ds.seen_items(u))} liked in train",
        )
    with c2:
        model_name = st.radio("Model", list(MODELS.keys()), horizontal=True)
    model = MODELS[model_name]

    left, right = st.columns([1, 1])
    with left:
        st.subheader("What they liked (train history)")
        seen = ds.seen_items(uidx)
        liked = pd.DataFrame({"Movie": [title(i) for i in seen]})
        st.dataframe(liked.head(15), use_container_width=True, hide_index=True)
        st.caption(f"{len(seen)} liked movies before the temporal cutoff.")

    with right:
        st.subheader(f"Top {K} recommendations — {model_name}")
        recs = model.recommend(uidx, k=K, exclude_seen=True)
        held = ds.test[uidx]
        rows = []
        for rank, item in enumerate(recs, 1):
            why = ""
            if model_name == "ItemKNN":
                src = model.explain(uidx, item)
                if src is not None:
                    why = f"because you liked “{title(src)}”"
            rows.append({
                "#": rank,
                "Movie": title(item),
                "Hit?": "✅" if item in held else "",
                "Why": why,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        n_hits = len(set(recs) & held)
        st.caption(f"✅ = one of this user's **actual future** picks. "
                   f"{n_hits}/{len(held)} held-out positives surfaced in the top {K}.")

# ------------------------------------------------------------------------- Leaderboard -----
with tab_board:
    st.subheader(f"Leaderboard — scored on {len(ds.test)} test users "
                 f"({ds.n_test_events:,} held-out future positives)")

    board = pd.DataFrame(LEADERBOARD).T[
        [f"NDCG@{K}", f"Recall@{K}", f"MAP@{K}", "Coverage", "Novelty(bits)"]
    ].round(4)
    st.dataframe(board, use_container_width=True)

    melted = board.reset_index().melt(id_vars="index", var_name="metric", value_name="value")
    fig = px.bar(melted, x="metric", y="value", color="index", barmode="group",
                 labels={"index": "model"}, title="Accuracy vs. beyond-accuracy by model")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
**How this is evaluated (and why it's honest)**

- **Implicit feedback:** a rating ≥ 4 is a positive; everything else is unobserved — the
  shape of real click/purchase data.
- **Global temporal split:** one cutoff in time. Train = the first 80% of interactions,
  test = the last 20%. *Every* training event precedes *every* test event, so no future
  co-occurrence leaks into the past (the #1 recsys eval mistake). The price is a smaller,
  honest test set of **{len(ds.test)} users** — a per-user split would inflate that but
  re-introduce the leak.
- **Ranking, not RMSE:** NDCG/Recall/MAP score the *order* of the top-{K}. **Coverage**
  (how much of the catalog is ever shown) and **Novelty** (−log₂ popularity, in bits) guard
  against a model that just re-serves blockbusters.
- **Read the result honestly:** the **Popularity** baseline is hard to beat on raw
  accuracy; **ItemKNN/ALS** trade a little accuracy for far higher coverage and novelty.
  Nothing here is tuned to make a favorite model win.
        """
    )
