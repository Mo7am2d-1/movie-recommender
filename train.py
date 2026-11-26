"""Movie recommender: item-item collaborative filtering (k-NN on ratings).

Dataset: MovieLens 100k (100,000 ratings from 943 users on 1682 movies).
Approach: build a user x movie rating matrix, then recommend movies whose
rating patterns are most similar (cosine distance) to movies the user
liked. Evaluation: held-out rating RMSE, compared against a
global-average baseline.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

# 1. Load
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
                     names=["item", "title"])
print(f"Loaded {len(ratings)} ratings, {ratings.user.nunique()} users, "
      f"{ratings.item.nunique()} movies")

# 2. Train/test split on ratings
train, test = train_test_split(ratings, test_size=0.2, random_state=42)
R = train.pivot_table(index="user", columns="item", values="rating").fillna(0)
print(f"Rating matrix: {R.shape}, density: {(R > 0).mean().mean():.1%}\n")

# 3. Item-item kNN with cosine distance on the centered matrix
item_mean = R.replace(0, np.nan).mean()
R_centered = R.sub(item_mean, axis=1).fillna(0)
knn = NearestNeighbors(n_neighbors=21, metric="cosine")
knn.fit(R_centered.T)

def predict_rating(user, item, k=20):
    """Weighted average of the user's ratings of the k most similar items."""
    if item not in R.columns or user not in R.index:
        return np.nan
    user_ratings = R.loc[user]
    rated = user_ratings[user_ratings > 0]
    if rated.empty:
        return np.nan
    dists, idxs = knn.kneighbors(R_centered[[item]].T, n_neighbors=len(R.columns))
    sims = 1 - dists[0]
    items = R.columns[idxs[0]]
    sim_items = [(i, s) for i, s in zip(items, sims) if i in rated.index and i != item][:k]
    if not sim_items:
        return np.nan
    num = sum(s * rated[i] for i, s in sim_items)
    den = sum(abs(s) for i, s in sim_items)
    return num / den if den else np.nan

# 4. Evaluate on held-out ratings (sample for speed) vs baseline
sample = test.sample(n=3000, random_state=42)
baseline = train["rating"].mean()
preds = [predict_rating(u, i) for u, i in zip(sample.user, sample.item)]
mask = ~np.isnan(preds)
rmse_model = mean_squared_error(sample.rating[mask], np.array(preds)[mask]) ** 0.5
rmse_base = mean_squared_error(sample.rating, [baseline] * len(sample)) ** 0.5
print(f"kNN rating prediction RMSE: {rmse_model:.3f} (on {mask.sum()} predictable ratings)")
print(f"Global-average baseline RMSE: {rmse_base:.3f}\n")

# 5. Demo: movies most similar to Star Wars
def similar_movies(item, n=10):
    dists, idxs = knn.kneighbors(R_centered[[item]].T, n_neighbors=n + 1)
    ids = [i for i in R.columns[idxs[0]] if i != item][:n]
    return titles.set_index("item").loc[ids, "title"].tolist()

sw = titles[titles.title.str.contains("Star Wars", case=False)].iloc[0]['item']
print(f"Movies most similar to '{titles.set_index('item').loc[sw, 'title']}':")
for t in similar_movies(sw):
    print(" -", t)

# 6. Plot: RMSE comparison + rating distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(["kNN model", "global average\nbaseline"], [rmse_model, rmse_base],
            color=["steelblue", "gray"])
axes[0].set_ylabel("RMSE on held-out ratings")
axes[0].set_title("Rating prediction error (lower is better)")
ratings.rating.value_counts().sort_index().plot.bar(ax=axes[1], color="steelblue")
axes[1].set_xlabel("rating")
axes[1].set_title("Rating distribution (MovieLens 100k)")
plt.tight_layout()
plt.savefig("results.svg")
print("Saved results.svg")
