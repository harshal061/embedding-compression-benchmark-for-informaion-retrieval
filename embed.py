import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
EMB_DIR = Path("embeddings")

DIMS = [128, 256, 384, 512, 768]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    EMB_DIR.mkdir(exist_ok=True)

    docs_768_file = EMB_DIR / "docs_768d.npy"
    queries_768_file = EMB_DIR / "queries_768d.npy"

    if docs_768_file.exists() and queries_768_file.exists():
        print("Embeddings already exist.")
        docs_768 = np.load(docs_768_file)
        queries_768 = np.load(queries_768_file)

    else:
        corpus = load_pickle(DATA_DIR / "corpus.pkl")
        queries = load_pickle(DATA_DIR / "queries.pkl")

        doc_texts = list(corpus.values())
        query_texts = list(queries.values())

        model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

        print("Encoding corpus...")
        docs_768 = model.encode(
            doc_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64
        )

        print("Encoding queries...")
        queries_768 = model.encode(
            query_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64
        )

        np.save(docs_768_file, docs_768)
        np.save(queries_768_file, queries_768)

    for dim in DIMS:
        np.save(
            EMB_DIR / f"docs_{dim}d.npy",
            docs_768[:, :dim]
        )

        np.save(
            EMB_DIR / f"queries_{dim}d.npy",
            queries_768[:, :dim]
        )

        print(f"Saved {dim}d embeddings")

    print("Finished.")