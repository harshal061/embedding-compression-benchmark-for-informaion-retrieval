import json
import pickle
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)

DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "scifact.zip"
EXTRACT_DIR = DATA_DIR / "scifact"


def download():
    DATA_DIR.mkdir(exist_ok=True)

    if ZIP_PATH.exists():
        print("Dataset already downloaded.")
        return

    print("Downloading SciFact...")

    response = requests.get(DATA_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(ZIP_PATH, "wb") as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

    print("Download complete.")


def unzip():
    if EXTRACT_DIR.exists():
        print("Dataset already extracted.")
        return

    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print("Extraction complete.")


def load_jsonl(path):
    records = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if "title" in row:
                text = f"{row.get('title', '')} {row.get('text', '')}"
            else:
                text = row["text"]

            records[str(row["_id"])] = text

    return records

def load_qrels(path):
    qrels = {}

    with open(path, "r", encoding="utf-8") as f:
        header = next(f)

        for line in f:
            parts = line.strip().split("\t")

            if len(parts) == 3:
                qid, doc_id, score = parts

            elif len(parts) == 4:
                qid, _, doc_id, score = parts

            else:
                continue

            if int(score) > 0:
                qrels.setdefault(str(qid), {})
                qrels[str(qid)][str(doc_id)] = 1

    return qrels

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":

    download()
    unzip()

    corpus_path = EXTRACT_DIR / "corpus.jsonl"
    query_path = EXTRACT_DIR / "queries.jsonl"
    qrels_path = EXTRACT_DIR / "qrels" / "test.tsv"

    print("Parsing corpus...")
    corpus = load_jsonl(corpus_path)

    print("Parsing queries...")
    queries = load_jsonl(query_path)

    print("Parsing qrels...")
    qrels = load_qrels(qrels_path)

    save_pickle(corpus, DATA_DIR / "corpus.pkl")
    save_pickle(queries, DATA_DIR / "queries.pkl")
    save_pickle(qrels, DATA_DIR / "qrels.pkl")

    print("\nDone!")
    print(f"Corpus documents : {len(corpus):,}")
    print(f"Queries          : {len(queries):,}")
    print(f"Qrel queries     : {len(qrels):,}")

    print("\nSaved:")
    print("  data/corpus.pkl")
    print("  data/queries.pkl")
    print("  data/qrels.pkl")