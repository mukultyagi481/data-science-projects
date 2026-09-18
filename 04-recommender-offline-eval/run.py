"""
Project 4 — Deciding what is worth A/B testing

Offline evaluation of three rankers on implicit feedback, with a time-based
split so no model ever sees the future:

  1. popularity            (the baseline any new model must beat)
  2. item-to-item cosine   (cheap, strong on dense histories)
  3. implicit-feedback ALS (matrix factorisation, Hu et al. 2008)

Metrics are NDCG@10, Recall@10, and catalogue coverage, each with a bootstrap
interval over users, plus a breakdown by how much history the user has. The
README turns the winner into an actual online test design.

Data: runs on simulated interactions by default. Drop MovieLens ratings.csv into
./data/ and pass --data data/ratings.csv to run on real data.

Usage:
    python run.py
    python run.py --data data/ratings.csv --max-users 20000
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(99)
OUT = os.path.dirname(os.path.abspath(__file__))
K = 10


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def simulate_interactions(n_users=8000, n_items=1500, rng=RNG):
    """Interactions with popularity skew, latent taste, and a time trend."""
    pop = rng.pareto(1.1, n_items) + 1
    pop = pop / pop.sum()
    user_taste = rng.normal(0, 1, (n_users, 3))
    item_taste = rng.normal(0, 1, (n_items, 3))
    activity = rng.integers(5, 60, n_users)

    rows = []
    for u in range(n_users):
        affinity = item_taste @ user_taste[u] / np.sqrt(3)
        score = 0.7 * np.log(pop) + 0.9 * affinity
        p = np.exp(score - score.max())
        p /= p.sum()
        items = rng.choice(n_items, size=activity[u], replace=False, p=p)
        # later interactions happen later in time, users staggered across window
        start = rng.integers(0, 700)
        ts = start + np.sort(rng.integers(0, 300, activity[u]))
        rows.append(pd.DataFrame({"user": u, "item": items, "ts": ts}))
    return pd.concat(rows, ignore_index=True)


def load_movielens(path, max_users=None, min_rating=4.0, rng=RNG):
    """MovieLens ratings.csv -> implicit feedback (a rating >= 4 counts as a like)."""
    df = pd.read_csv(path, usecols=["userId", "movieId", "rating", "timestamp"])
    df = df[df.rating >= min_rating]
    if max_users:
        keep = rng.choice(df.userId.unique(), size=min(max_users, df.userId.nunique()),
                          replace=False)
        df = df[df.userId.isin(keep)]
    return df.rename(columns={"userId": "user", "movieId": "item", "timestamp": "ts"})[
        ["user", "item", "ts"]]


def time_split(df, test_frac=0.2):
    """Global time cutoff: train on the past, evaluate on the future."""
    cutoff = df.ts.quantile(1 - test_frac)
    train = df[df.ts <= cutoff]
    test = df[df.ts > cutoff]
    # keep only users and items the training set has seen
    test = test[test.user.isin(train.user.unique()) & test.item.isin(train.item.unique())]
    return train, test, cutoff


def to_matrix(train):
    users = np.sort(train.user.unique())
    items = np.sort(train.item.unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {it: i for i, it in enumerate(items)}
    rows = train.user.map(u_idx).values
    cols = train.item.map(i_idx).values
    mat = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                        shape=(len(users), len(items)))
    mat.data[:] = 1.0
    return mat, u_idx, i_idx


# ----------------------------------------------------------------------------
# Models: each returns a score matrix for the requested users
# ----------------------------------------------------------------------------
def popularity_scores(mat, user_rows):
    pop = np.asarray(mat.sum(axis=0)).ravel()
    return np.tile(pop, (len(user_rows), 1))


def item_knn_scores(mat, user_rows, neighbours=50):
    """Cosine similarity between item columns, truncated to top neighbours."""
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=0)).ravel()) + 1e-9
    normed = mat.multiply(sp.csr_matrix(1.0 / norms))
    sim = (normed.T @ normed).toarray()
    np.fill_diagonal(sim, 0.0)
    if neighbours < sim.shape[0]:
        cut = np.partition(sim, -neighbours, axis=1)[:, -neighbours][:, None]
        sim = np.where(sim >= cut, sim, 0.0)
    return mat[user_rows].toarray() @ sim


def als_fit(mat, factors=32, iters=15, reg=0.05, alpha=20.0, rng=RNG):
    """Implicit-feedback ALS. Confidence c = 1 + alpha * interaction."""
    n_u, n_i = mat.shape
    X = 0.01 * rng.normal(size=(n_u, factors))
    Y = 0.01 * rng.normal(size=(n_i, factors))
    eye = reg * np.eye(factors)
    csr, csc = mat.tocsr(), mat.tocsc()

    def solve(fixed, sparse_mat, n_rows):
        out = np.zeros((n_rows, factors))
        gram = fixed.T @ fixed
        for r in range(n_rows):
            idx = sparse_mat.indices[sparse_mat.indptr[r]:sparse_mat.indptr[r + 1]]
            if idx.size == 0:
                continue
            f = fixed[idx]
            a = gram + alpha * (f.T @ f) + eye
            b = (1 + alpha) * f.sum(axis=0)
            out[r] = np.linalg.solve(a, b)
        return out

    for _ in range(iters):
        X = solve(Y, csr, n_u)
        Y = solve(X, csc.T.tocsr(), n_i)
    return X, Y


def als_scores(X, Y, user_rows):
    return X[user_rows] @ Y.T


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def ndcg_recall(scores, mat, user_rows, truth, k=K):
    """Rank top-k unseen items per user; compare against held-out items."""
    seen = mat[user_rows].toarray() > 0
    scores = np.where(seen, -np.inf, scores)
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)

    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ndcgs, recalls = [], []
    for row, recs in enumerate(top):
        rel = truth[row]
        if not rel:
            continue
        hits = np.array([1.0 if i in rel else 0.0 for i in recs])
        ideal = discount[:min(len(rel), k)].sum()
        ndcgs.append((hits * discount).sum() / ideal)
        recalls.append(hits.sum() / min(len(rel), k))
    coverage = len(np.unique(top)) / mat.shape[1]
    return np.array(ndcgs), np.array(recalls), coverage, top


def bootstrap_ci(values, reps=1000, rng=RNG):
    if values.size == 0:
        return (np.nan, np.nan)
    draws = rng.choice(values, size=(reps, values.size), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def make_figure(results, by_history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    names = [r["model"] for r in results]
    means = [r["ndcg@10"] for r in results]
    err = [[r["ndcg@10"] - r["ndcg_ci"][0] for r in results],
           [r["ndcg_ci"][1] - r["ndcg@10"] for r in results]]
    axes[0].bar(names, means, yerr=err, color=["#8a96a3", "#5d6b73", "#0e8c6a"], capsize=4)
    axes[0].set_ylabel("NDCG@10")
    axes[0].set_title("Ranking quality, with bootstrap intervals")

    for model, sub in by_history.groupby("model"):
        axes[1].plot(sub.history_bucket, sub["ndcg@10"], "o-", label=model)
    axes[1].set_xlabel("items in the user's training history")
    axes[1].set_ylabel("NDCG@10")
    axes[1].set_title("Where the gain actually comes from")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures/offline_evaluation.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="MovieLens ratings.csv")
    ap.add_argument("--max-users", type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    df = load_movielens(args.data, args.max_users) if args.data else simulate_interactions()
    train, test, cutoff = time_split(df)
    mat, u_idx, i_idx = to_matrix(train)

    test_users = np.sort(test.user.unique())
    user_rows = np.array([u_idx[u] for u in test_users])
    truth = [set(test.loc[test.user == u, "item"].map(i_idx).dropna().astype(int))
             for u in test_users]
    history = np.asarray(mat[user_rows].sum(axis=1)).ravel()

    X, Y = als_fit(mat)
    model_scores = {
        "popularity": popularity_scores(mat, user_rows),
        "item-item cosine": item_knn_scores(mat, user_rows),
        "ALS (32 factors)": als_scores(X, Y, user_rows),
    }

    results, hist_rows = [], []
    for name, scores in model_scores.items():
        ndcgs, recalls, coverage, _ = ndcg_recall(scores, mat, user_rows, truth)
        results.append({
            "model": name,
            "ndcg@10": float(ndcgs.mean()),
            "ndcg_ci": [float(x) for x in bootstrap_ci(ndcgs)],
            "recall@10": float(recalls.mean()),
            "recall_ci": [float(x) for x in bootstrap_ci(recalls)],
            "catalogue_coverage": float(coverage),
            "users_evaluated": int(ndcgs.size),
        })
        buckets = pd.cut(history, [0, 10, 20, 40, np.inf],
                         labels=["1-10", "11-20", "21-40", "40+"])
        per_user = pd.DataFrame({"bucket": buckets[:ndcgs.size], "ndcg": ndcgs})
        for b, sub in per_user.groupby("bucket", observed=True):
            hist_rows.append({"model": name, "history_bucket": str(b),
                              "ndcg@10": float(sub.ndcg.mean())})

    by_history = pd.DataFrame(hist_rows)
    by_history.to_csv(os.path.join(OUT, "results/by_history.csv"), index=False)
    pd.DataFrame(results).to_csv(os.path.join(OUT, "results/offline_metrics.csv"), index=False)
    make_figure(results, by_history)

    base = next(r for r in results if r["model"] == "popularity")
    best = max(results, key=lambda r: r["ndcg@10"])
    summary = {
        "data": args.data or "simulated",
        "train_interactions": int(len(train)),
        "test_interactions": int(len(test)),
        "users_evaluated": best["users_evaluated"],
        "results": results,
        "best_model": best["model"],
        "ndcg_lift_over_popularity_pct": round(
            100 * (best["ndcg@10"] - base["ndcg@10"]) / base["ndcg@10"], 1),
        "coverage_multiple_over_popularity": round(
            best["catalogue_coverage"] / base["catalogue_coverage"], 2),
    }
    with open(os.path.join(OUT, "results/summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(by_history.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
