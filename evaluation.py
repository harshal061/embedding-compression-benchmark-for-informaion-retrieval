import json
import pickle
from pathlib import Path



import numpy as np
from tqdm import tqdm

DATA_DIR = Path("data")
EMB_DIR = Path("embeddings")

DIMS = [128, 256, 384, 512, 768]
TOP_K = 10
FULL_DIM = 768

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + 1e-12)


def dcg(relevances, k):
    relevances = np.asarray(relevances[:k])

    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(relevances, k):
    actual = dcg(relevances, k)

    ideal = dcg(
        sorted(relevances, reverse=True),
        k
    )

    if ideal == 0:
        return 0.0

    return actual / ideal


def mrr(relevances):
    for rank, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / rank
    return 0.0

def recall_at_k(relevances, total_relevant, k):
    if total_relevant == 0:
        return 0.0

    retrieved_relevant = np.sum(relevances[:k])

    return retrieved_relevant / total_relevant

if __name__ == "__main__":
    corpus = load_pickle(DATA_DIR / "corpus.pkl")
    queries = load_pickle(DATA_DIR / "queries.pkl")
    qrels = load_pickle(DATA_DIR / "qrels.pkl")

    doc_ids = list(corpus.keys())
    query_ids = list(queries.keys())

    results = {}

    for dim in DIMS:
        print(f"\nEvaluating {dim}d")

        docs = np.load(
            EMB_DIR / f"docs_{dim}d.npy"
        )

        query_embs = np.load(
            EMB_DIR / f"queries_{dim}d.npy"
        )

        docs = normalize(docs)
        query_embs = normalize(query_embs)

        sim = query_embs @ docs.T

        ndcg_scores = []
        mrr_scores = []
        recall_scores = []

        for q_idx in tqdm(range(len(query_ids))):
            qid = query_ids[q_idx]

            scores = sim[q_idx]

            top_idx = np.argsort(scores)[::-1][:TOP_K]

            rels = []

            relevant_docs = qrels.get(qid, {})

            for idx in top_idx:
                doc_id = doc_ids[idx]
                rels.append(
                    1 if doc_id in relevant_docs else 0
                )

            total_relevant = len(relevant_docs)

            ndcg_scores.append(
                ndcg_at_k(rels, TOP_K)
            )

            mrr_scores.append(
                mrr(rels)
            )

            recall_scores.append(
                recall_at_k(
                    rels,
                    total_relevant,
                    TOP_K
                )
            )

        results[str(dim)] = {
            "ndcg@10": float(np.mean(ndcg_scores)),
            "mrr": float(np.mean(mrr_scores)),
            "recall@10": float(np.mean(recall_scores)),
            "compression_ratio": dim / FULL_DIM
        }

    print("| Dimension | Compression | nDCG@10 | Recall@10 | MRR |")
    print("|-----------|-------------|----------|-----------|------|")

    for dim in DIMS:
        r = results[str(dim)]

        print(
            f"| {dim} | "
            f"{r['compression_ratio']:.4f} | "
            f"{r['ndcg@10']:.4f} | "
            f"| {r['recall@10']:.4f} "
            f"{r['mrr']:.4f} |"
        )

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results.json")