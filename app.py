"""Streamlit app: movie recommender (MovieLens 100k).

Item-item collaborative filtering: movies are similar if users rated them
similarly (cosine distance on the centered rating matrix). Pick a movie
to see its most similar movies, or enter a user id (1-943) to get
personal top-10 recommendations.

Run with:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def build_model():
    import os, zipfile, urllib.request
    if not os.path.exists("u.data"):
        print("MovieLens data not found, downloading ...")
        urllib.request.urlretrieve(
            "https://files.grouplens.org/datasets/movielens/ml-100k.zip", "ml.zip")
        with zipfile.ZipFile("ml.zip") as z:
            z.extract("ml-100k/u.data", ".")
            z.extract("ml-100k/u.item", ".")
        os.replace("ml-100k/u.data", "u.data")
        os.replace("ml-100k/u.item", "u.item")
        os.rmdir("ml-100k")
        os.remove("ml.zip")

    ratings = pd.read_csv("u.data", sep="\t", names=["user", "item", "rating", "ts"])
    titles = pd.read_csv("u.item", sep="|", encoding="latin-1", usecols=[0, 1],
                         names=["item", "title"]).set_index("item")["title"]
    R = ratings.pivot_table(index="user", columns="item", values="rating").fillna(0)
    item_mean = R.replace(0, np.nan).mean().fillna(3.0)
    Rc = R.sub(item_mean, axis=1).fillna(0)
    sim = cosine_similarity(Rc.T)
    sim = pd.DataFrame(sim, index=R.columns, columns=R.columns)
    np.fill_diagonal(sim.values, 0)
    return R, Rc, sim, titles, ratings


R, Rc, sim, titles, ratings = build_model()

st.title("Movie recommender")
st.write("Practice project on MovieLens 100k "
         f"({len(ratings)} ratings, {R.shape[0]} users, {R.shape[1]} movies). "
         "Item-item collaborative filtering with cosine similarity.")

tab1, tab2 = st.tabs(["Similar movies", "Recommendations for a user"])

with tab1:
    options = sorted(titles.dropna().unique())
    pick = st.selectbox("Pick a movie", options,
                        index=int(np.where(np.array(options) == "Star Wars (1977)")[0][0]))
    item_id = titles[titles == pick].index[0]
    top = sim[item_id].nlargest(10)
    st.subheader(f"Movies most similar to {pick}")
    st.table(pd.DataFrame({
        "movie": [titles.get(i, str(i)) for i in top.index],
        "similarity": top.values.round(3)}))

with tab2:
    uid = st.number_input("User id (1-943)", min_value=1, max_value=int(R.index.max()), value=1)
    if uid in R.index:
        seen = R.loc[uid] > 0
        # score unseen items by similarity-weighted sum of the user's centered ratings
        scores = sim.values @ Rc.loc[uid].values
        scores = pd.Series(scores, index=sim.index)
        scores[seen] = -np.inf
        top = scores.nlargest(10)
        st.subheader(f"Top 10 recommendations for user {uid}")
        st.table(pd.DataFrame({
            "movie": [titles.get(i, str(i)) for i in top.index],
            "score": top.values.round(2)}))
        st.caption("Score = sum of similarity x centered rating over the movies the user rated.")
    else:
        st.info("That user id has no ratings in the dataset.")
