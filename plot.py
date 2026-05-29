import json
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
)

from tqdm import tqdm

DATA_DIR = Path("data")
EMB_DIR = Path("embeddings")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + 1e-12)


if __name__ == "__main__":
    corpus = load_pickle(DATA_DIR / "corpus.pkl")
    queries = load_pickle(DATA_DIR / "queries.pkl")
    qrels = load_pickle(DATA_DIR / "qrels.pkl")

    docs = np.load(
        EMB_DIR / "docs_768d.npy"
    )

    query_embs = np.load(
        EMB_DIR / "queries_768d.npy"
    )

    docs = normalize(docs)
    query_embs = normalize(query_embs)

    doc_ids = list(corpus.keys())
    query_ids = list(queries.keys())

    rows = []

    NEGATIVE_RATIO = 5

    for q_idx, qid in enumerate(tqdm(query_ids)):

        qvec = query_embs[q_idx]

        scores = docs @ qvec

        relevant_docs = set(
            qrels.get(qid, {}).keys()
        )

        qtext = queries[qid]

        # positives
        for doc_id in relevant_docs:

            d_idx = doc_ids.index(doc_id)

            rows.append({
                "cosine_score": float(scores[d_idx]),
                "query_len": len(qtext),
                "doc_len": len(corpus[doc_id]),
                "label": 1
            })

        # negatives
        candidates = list(
            set(doc_ids) - relevant_docs
        )

        negatives = random.sample(
            candidates,
            min(
                len(candidates),
                NEGATIVE_RATIO *
                max(1, len(relevant_docs))
            )
        )

        for doc_id in negatives:

            d_idx = doc_ids.index(doc_id)

            rows.append({
                "cosine_score": float(scores[d_idx]),
                "query_len": len(qtext),
                "doc_len": len(corpus[doc_id]),
                "label": 0
            })

    df = pd.DataFrame(rows)

    X = df[
        ["cosine_score",
         "query_len",
         "doc_len"]
    ]

    y = df["label"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            preds
        )
    )

    with open("results.json", "r") as f:
        results = json.load(f)

    dims = sorted(
        [int(x) for x in results.keys()]
    )

    ndcg = [
        results[str(d)]["ndcg@10"]
        for d in dims
    ]

    mrr = [
        results[str(d)]["mrr"]
        for d in dims
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].plot(
        dims,
        ndcg,
        marker="o"
    )

    axes[0].set_title("nDCG@10")
    axes[0].set_xlabel("Dimensions")
    axes[0].set_ylabel("Score")

    axes[1].bar(
        [str(d) for d in dims],
        mrr
    )

    axes[1].set_title("MRR")
    axes[1].set_xlabel("Dimensions")
    axes[1].set_ylabel("Score")

    fig.suptitle(
        "SciFact Retrieval vs. Embedding Dimensionality"
    )

    plt.tight_layout()

    plt.savefig(
        "truncationbench.png",
        dpi=300,
        bbox_inches="tight"
    )

    print("Saved truncationbench.png")